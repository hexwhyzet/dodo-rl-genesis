import math
import os

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../../assets/robots/dodo")


class DodoEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, enable_render=False):
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
                camera_pos=(2.5, 0.0, 1.2),
                camera_lookat=(0.0, 0.0, 0.4),
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

        self.render_cameras = []
        if enable_render:
            self.render_cameras = [
                self.scene.add_camera(
                    res=(1280, 720),
                    pos=(1.5, 0.0, 0.7),
                    lookat=(0.0, 0.0, 0.3),
                    fov=40,
                    GUI=False,
                ),
                self.scene.add_camera(
                    res=(1280, 720),
                    pos=(0.0, 1.5, 0.7),
                    lookat=(0.0, 0.0, 0.3),
                    fov=40,
                    GUI=False,
                ),
            ]

        self.scene.build(n_envs=num_envs)

        self.show_viewer = show_viewer
        self._cmd_arrow_node = None
        if show_viewer:
            self.scene.viewer.follow_entity(self.robot, smoothing=0.5)
        for cam in self.render_cameras:
            cam.follow_entity(self.robot, smoothing=0.5)

        # ------------------------------------------------------------------ link indices
        link_names = [link.name for link in self.robot.links]
        self.non_foot_link_indices = torch.tensor(
            [i for i, name in enumerate(link_names)
             if name not in ("foot_right", "foot_left")],
            dtype=torch.long, device=gs.device,
        )
        self.foot_right_link = self.robot.get_link("foot_right")
        self.foot_left_link  = self.robot.get_link("foot_left")
        self.foot_link_indices = torch.tensor(
            [link_names.index("foot_right"), link_names.index("foot_left")],
            dtype=torch.long, device=gs.device,
        )

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
        self.kp = torch.tensor(kp_list, dtype=gs.tc_float, device=gs.device)
        self.kd = torch.tensor(kd_list, dtype=gs.tc_float, device=gs.device)

        # joint limits from URDF [hip, upper, lower, foot] × 2
        self.dof_limits_lower = torch.tensor(
            [-0.35, -1.57, -3.1416, -1.05, -0.35, -1.57, -3.1416, -1.05],
            dtype=gs.tc_float, device=gs.device,
        )
        self.dof_limits_upper = torch.tensor(
            [0.35, 1.57, 1.3963, 1.57, 0.35, 1.57, 1.3963, 1.57],
            dtype=gs.tc_float, device=gs.device,
        )

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
        n_links = len(self.robot.links)
        self.contact_forces      = torch.zeros((num_envs, n_links, 3), dtype=gs.tc_float, device=gs.device)
        self.last_contact_forces = torch.zeros((num_envs, n_links, 3), dtype=gs.tc_float, device=gs.device)
        self.torques = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.extras: dict = {}

        # ------------------------------------------------------------------ frame stacking
        self.obs_history_len = 1  # no history, obs size stays 33
        self.single_obs_dim = 33  # ang_vel(3) + gravity(3) + commands(3) + dof_pos(8) + dof_vel(8) + actions(8)
        self.obs_history = torch.zeros(
            (num_envs, self.obs_history_len, self.single_obs_dim), dtype=gs.tc_float, device=gs.device
        )

        # ------------------------------------------------------------------ domain randomization
        self.base_kp = torch.tensor(kp_list, dtype=gs.tc_float, device=gs.device)
        self.base_kd = torch.tensor(kd_list, dtype=gs.tc_float, device=gs.device)
        self.push_interval = 0  # 0 = disabled; enable after robot learns to walk (~1000 iters)
        # per-env push strength seeded by gs.init(seed=...) for reproducibility
        self.push_strength = torch.rand(num_envs, device=gs.device) * 40.0 + 10.0  # 10-50 N

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
        # 20% of resamples are stand-still commands so the robot learns to stand quietly
        stand_mask = torch.rand(self.num_envs, device=gs.device) < 0.2
        commands[stand_mask] = 0.0
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

        if self.show_viewer or self.render_cameras:
            self._draw_command_arrow()

    def _draw_command_arrow(self):
        try:
            pos = self.base_pos[0].cpu().numpy()

            # use world-frame command if available (set by gamepad), else convert from body
            if hasattr(self, '_world_cmd'):
                world_vx, world_vy = self._world_cmd
            else:
                cmd = self.commands[0].cpu().numpy()
                q   = self.base_quat[0].cpu().numpy()
                vx, vy = float(cmd[0]), float(cmd[1])
                yaw = math.atan2(2.0 * (q[0]*q[3] + q[1]*q[2]),
                                 1.0 - 2.0 * (q[2]**2 + q[3]**2))
                world_vx = vx * math.cos(yaw) - vy * math.sin(yaw)
                world_vy = vx * math.sin(yaw) + vy * math.cos(yaw)

            speed = math.sqrt(world_vx**2 + world_vy**2)

            ctx = self.scene._visualizer.context
            if self._cmd_arrow_node is not None:
                ctx.clear_external_node(self._cmd_arrow_node)
                self._cmd_arrow_node = None

            if speed < 0.05:
                return

            scale = 0.5  # 0.5m per 1 m/s — arrow length proportional to speed
            vec = np.array([world_vx * scale, world_vy * scale, 0.0], dtype=np.float32)
            arrow_pos = np.array([pos[0], pos[1], pos[2] + 0.3], dtype=np.float32)

            self._cmd_arrow_node = self.scene.draw_debug_arrow(
                pos=arrow_pos, vec=vec, radius=0.012, color=(0.15, 0.9, 0.4, 0.85),
            )
        except Exception as e:
            print(f"[arrow] ERROR: {e}")

    # ---------------------------------------------------------------------- step / reset

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 6 + self.num_actions))

        # random push perturbation — applies force to base link every push_interval steps
        if self.push_interval > 0:
            push_mask = (self.episode_length_buf % self.push_interval == 0) & (self.episode_length_buf > 0)
        if self.push_interval > 0 and push_mask.any():
            push_idx = torch.where(push_mask)[0]
            direction = torch.nn.functional.normalize(
                torch.randn(len(push_idx), 3, device=gs.device) * torch.tensor([1.0, 1.0, 0.0], device=gs.device),
                dim=1,
            )
            force = direction * self.push_strength[push_idx, None]
            self.scene.rigid_solver.apply_links_external_force(
                force=force.unsqueeze(1),
                links_idx=[self.robot.base_link_idx],
                envs_idx=push_idx,
            )

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

        pos_error = self.actions * self.env_cfg["action_scale"] - (self.dof_pos - self.default_dof_pos)
        self.torques = self.kp * pos_error - self.kd * self.dof_vel

        self.last_contact_forces.copy_(self.contact_forces)
        self.contact_forces = self.robot.get_links_net_contact_force()

        # termination computed BEFORE rewards so fall_penalty fires correctly
        timed_out = self.episode_length_buf > self.max_episode_length
        fell = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | self.scene.rigid_solver.get_error_envs_mask()
        )
        non_foot_forces = self.contact_forces[:, self.non_foot_link_indices, :]
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
        # domain randomization: PD gains ±20%, friction 0.5-1.5x, push strength 20-100N
        if envs_idx is None:
            n_reset = self.num_envs
            dr_idx = None
        else:
            n_reset = int(envs_idx.sum().item())
            dr_idx = torch.where(envs_idx)[0]

        if n_reset > 0:
            # PD gains: Genesis only supports global (1D) kp/kd, randomize on full reset
            if dr_idx is None:
                kp_rand = self.base_kp * (0.8 + 0.4 * torch.rand(self.num_actions, device=gs.device))
                kd_rand = self.base_kd * (0.8 + 0.4 * torch.rand(self.num_actions, device=gs.device))
                self.robot.set_dofs_kp(kp_rand, self.motors_dof_idx)
                self.robot.set_dofs_kv(kd_rand, self.motors_dof_idx)

            # friction randomization disabled — hurts early learning; enable after robot can walk
            # n_links = len(self.robot.links)
            # friction_ratio = 0.5 + torch.rand(n_reset, n_links, device=gs.device)
            # self.robot.set_friction_ratio(friction_ratio, envs_idx=dr_idx)

            new_push = torch.rand(n_reset, device=gs.device) * 80.0 + 20.0
            if dr_idx is None:
                self.push_strength.copy_(new_push)
            else:
                self.push_strength[dr_idx] = new_push

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

        # reset obs history for done envs
        if envs_idx is None:
            self.obs_history.zero_()
        else:
            self.obs_history.masked_fill_(envs_idx[:, None, None], 0.0)

        self._resample_commands(envs_idx)

    def _update_observation(self):
        current_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],               # 3
                self.projected_gravity,                                        # 3
                self.commands * self.commands_scale,                           # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 8
                self.dof_vel * self.obs_scales["dof_vel"],                    # 8
                self.actions,                                                  # 8
            ),
            dim=-1,
        )  # (num_envs, 33)
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = current_obs
        self.obs_buf = self.obs_history.reshape(self.num_envs, -1)  # (num_envs, 33*5=165)

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
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_base_ang_vel(self):
        return torch.sum(torch.square(self.base_ang_vel), dim=1)

    def _reward_lateral_motion(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_mechanical_power(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _foot_contacts(self):
        foot_forces = self.contact_forces[:, self.foot_link_indices, :]
        r = torch.norm(foot_forces[:, 0, :], dim=-1) > 1.0
        l = torch.norm(foot_forces[:, 1, :], dim=-1) > 1.0
        return r, l

    def _reward_foot_slip(self):
        r_contact, l_contact = self._foot_contacts()
        slip_r = torch.norm(self.foot_right_link.get_vel()[:, :2], dim=1) * r_contact.float()
        slip_l = torch.norm(self.foot_left_link.get_vel()[:, :2], dim=1) * l_contact.float()
        return slip_r + slip_l

    def _reward_feet_impact(self):
        curr = self.contact_forces[:, self.foot_link_indices, :]
        prev = self.last_contact_forces[:, self.foot_link_indices, :]
        delta = torch.norm(curr - prev, dim=-1)
        return torch.sum(delta, dim=1)

    def _reward_step_length(self):
        r_contact, l_contact = self._foot_contacts()
        fwd_r = self.foot_right_link.get_vel()[:, 0] * (~r_contact).float()
        fwd_l = self.foot_left_link.get_vel()[:, 0]  * (~l_contact).float()
        return torch.clamp(fwd_r + fwd_l, min=0.0)

    def _reward_gait_alternation(self):
        r_contact, l_contact = self._foot_contacts()
        alternating = (r_contact ^ l_contact).float()
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        weight = torch.clamp(cmd_speed / 0.3, 0.0, 1.0)
        return alternating * weight

    def _reward_double_stance(self):
        r_contact, l_contact = self._foot_contacts()
        both = (r_contact & l_contact).float()
        speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return both * torch.clamp(speed - 0.3, min=0.0)

    def _reward_excessive_flight(self):
        r_contact, l_contact = self._foot_contacts()
        return (~r_contact & ~l_contact).float()

    def _reward_joint_limits(self):
        margin = 0.1
        below = torch.clamp(self.dof_limits_lower + margin - self.dof_pos, min=0.0)
        above = torch.clamp(self.dof_pos - (self.dof_limits_upper - margin), min=0.0)
        return torch.sum(below + above, dim=1)

    def _reward_gait_symmetry(self):
        # sagittal joints should be anti-phase: right ≈ -left
        right = self.dof_pos[:, 1:4]  # upper, lower, foot right
        left  = self.dof_pos[:, 5:8]  # upper, lower, foot left
        anti_sym_err = torch.sum(torch.square(right + left), dim=1)
        return torch.exp(-anti_sym_err / 0.5)

    def _reward_heading(self):
        # reward aligning body heading with world-frame velocity direction
        # only meaningful when actually moving
        world_vel = self.robot.get_vel()[:, :2]  # (n_envs, 2) world XY velocity
        speed = torch.norm(world_vel, dim=1)

        # robot forward direction in world: rotate body X=[1,0,0] by base_quat
        body_x = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        body_x[:, 0] = 1.0
        world_forward = transform_by_quat(body_x, self.base_quat)[:, :2]

        # cos(angle between velocity and heading): 1=aligned, -1=opposite
        vel_dir = world_vel / (speed.unsqueeze(1) + 1e-6)
        fwd_norm = world_forward / (torch.norm(world_forward, dim=1, keepdim=True) + 1e-6)
        alignment = torch.sum(vel_dir * fwd_norm, dim=1)

        # only apply when moving fast enough to have a meaningful direction
        return torch.where(speed > 0.2, alignment, torch.zeros_like(speed))

    def _reward_unallowed_contacts(self):
        non_foot_forces = self.contact_forces[:, self.non_foot_link_indices, :]
        return torch.any(torch.norm(non_foot_forces, dim=-1) > 1.0, dim=1).float()
