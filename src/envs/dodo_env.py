import math
import os

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../../assets/robots/dodo")


class DodoEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs: int = num_envs
        self.num_actions: int = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands: int = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency: bool = env_cfg.get("simulate_action_latency", True)
        self.dt: float = 0.02  # 50 Hz control
        self.max_episode_length: int = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict = obs_cfg["obs_scales"]
        self.reward_scales: dict = reward_cfg["reward_scales"]

        # ------------------------------------------------------------------ scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=True,
                tolerance=1e-5,
                max_collision_pairs=64,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(ASSETS_DIR, "urdf/dodo_daimao.urdf"),
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            )
        )

        self.scene.build(n_envs=num_envs)

        # ------------------------------------------------------------------ link indices
        # Genesis merges fixed joints — foot_sole is merged into foot_right/foot_left
        # Only foot_right and foot_left are allowed to touch the ground
        link_names = [link.name for link in self.robot.links]
        self.non_foot_link_indices = torch.tensor(
            [i for i, name in enumerate(link_names)
             if name not in ("foot_right", "foot_left")],
            dtype=torch.long, device=gs.device,
        )
        self.foot_right_link = self.robot.get_link("foot_right")
        self.foot_left_link  = self.robot.get_link("foot_left")

        # ------------------------------------------------------------------ DOF indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD gains — per-joint kp/kd from config
        kp_list = env_cfg["kp"] if isinstance(env_cfg["kp"], list) else [env_cfg["kp"]] * self.num_actions
        kd_list = env_cfg["kd"] if isinstance(env_cfg["kd"], list) else [env_cfg["kd"]] * self.num_actions
        self.robot.set_dofs_kp(kp_list, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd_list, self.motors_dof_idx)

        # ------------------------------------------------------------------ constants
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )

        # ------------------------------------------------------------------ buffers
        self.base_lin_vel = torch.empty((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.commands_limits: tuple = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                command_cfg["lin_vel_x_range"],
                command_cfg["lin_vel_y_range"],
                command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.extras: dict = {}

        # ------------------------------------------------------------------ rewards
        self.reward_functions, self.episode_sums = {}, {}
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    # ---------------------------------------------------------------------- commands

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    # ---------------------------------------------------------------------- step / reset

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 6 + self.num_actions))
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # termination computed BEFORE rewards so fall_penalty fires correctly
        timed_out = self.episode_length_buf > self.max_episode_length
        fell = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | self.scene.rigid_solver.get_error_envs_mask()
        )
        contact_forces = self.robot.get_links_net_contact_force()
        non_foot_forces = contact_forces[:, self.non_foot_link_indices, :]
        bad_contact = torch.any(torch.norm(non_foot_forces, dim=-1) > 1.0, dim=1)

        # self-collision: any link touching another link of the same robot
        self_contacts = self.robot.get_contacts(with_entity=self.robot)
        self_collision = self_contacts["valid_mask"].any(dim=-1)  # (n_envs,)

        self.reset_buf = timed_out | fell | bad_contact | self_collision
        self.extras["time_outs"] = timed_out.to(dtype=gs.tc_float)

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],               # 3
                self.projected_gravity,                                        # 3
                self.commands * self.commands_scale,                           # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 8
                self.dof_vel * self.obs_scales["dof_vel"],                    # 8
                self.actions,                                                  # 8
            ),
            dim=-1,
        )  # total: 33

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ---------------------------------------------------------------------- rewards

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_upright(self):
        # 1.0 when perfectly vertical, 0.0 when horizontal
        return 1.0 - torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_feet_orientation(self):
        # penalize feet not being parallel to the ground
        # project gravity into each foot frame — should be [0,0,-1] when foot is level
        foot_r_quat = self.foot_right_link.get_quat()  # (n_envs, 4)
        foot_l_quat = self.foot_left_link.get_quat()
        grav_in_foot_r = transform_by_quat(self.global_gravity, inv_quat(foot_r_quat))
        grav_in_foot_l = transform_by_quat(self.global_gravity, inv_quat(foot_l_quat))
        # xy components should be zero when foot is horizontal
        err_r = torch.sum(torch.square(grav_in_foot_r[:, :2]), dim=1)
        err_l = torch.sum(torch.square(grav_in_foot_l[:, :2]), dim=1)
        return err_r + err_l

    def _reward_alive(self):
        # +1 every step the robot hasn't fallen
        return (~self.reset_buf).float()

    def _reward_episode_success(self):
        # one-time bonus for surviving the full episode (timeout, not a fall)
        timed_out = self.extras.get("time_outs", torch.zeros_like(self.reset_buf)).bool()
        return timed_out.float()

    def _reward_fall_penalty(self):
        # -1 on any termination that is not a timeout (fell or bad contact)
        timed_out = self.extras.get("time_outs", torch.zeros_like(self.reset_buf)).bool()
        return (self.reset_buf & ~timed_out).float()

    def _reward_torques(self):
        # PD torque estimate: kp * pos_error - kd * vel
        kp = torch.tensor(self.env_cfg["kp"], dtype=gs.tc_float, device=gs.device)
        kd = torch.tensor(self.env_cfg["kd"], dtype=gs.tc_float, device=gs.device)
        pos_error = self.actions * self.env_cfg["action_scale"] - (self.dof_pos - self.default_dof_pos)
        torques = kp * pos_error - kd * self.dof_vel
        return torch.sum(torch.square(torques), dim=1)

    def _reward_dof_acc(self):
        # penalize joint accelerations
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_base_ang_vel(self):
        # penalize any body rotation (want to stand still)
        return torch.sum(torch.square(self.base_ang_vel), dim=1)

    def _reward_unallowed_contacts(self):
        # penalize any contact force on non-foot_sole links
        forces = self.robot.get_links_net_contact_force()  # (n_envs, n_links, 3)
        non_foot_forces = forces[:, self.non_foot_link_indices, :]
        return torch.any(torch.norm(non_foot_forces, dim=-1) > 1.0, dim=1).float()
