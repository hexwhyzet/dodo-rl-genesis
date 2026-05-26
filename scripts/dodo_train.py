#!/usr/bin/env python3
import argparse
import os
import pickle
import shutil
from pathlib import Path
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.envs.dodo_env import DodoEnv
from src.streaming.video_recorder import VideoRecorder


def get_train_cfg(exp_name):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 48,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs():
    env_cfg = {
        "num_actions": 8,
        # joint order matches joint_names_dodo_daimao.yaml (excluding empty first entry)
        "joint_names": [
            "hip_right",
            "upper_leg_right",
            "lower_leg_right",
            "foot_right",
            "hip_left",
            "upper_leg_left",
            "lower_leg_left",
            "foot_left",
        ],
        # default standing pose — vertical legs, slight knee bend for compliance
        "default_joint_angles": {
            "hip_right":       0.0,
            "upper_leg_right": 1.32,
            "lower_leg_right": -2.08,
            "foot_right":      1.25,
            "hip_left":        0.0,
            "upper_leg_left":  1.32,
            "lower_leg_left":  -2.08,
            "foot_left":       1.25,
        },
        # PD gains — hips stronger, knees/ankles lighter
        "kp": [20.0, 20.0, 10.0, 10.0, 20.0, 20.0, 10.0, 10.0],
        "kd": [0.5,  0.5,  0.3,  0.3,  0.5,  0.5,  0.3,  0.3],
        # termination
        "termination_if_roll_greater_than":  30,  # degrees
        "termination_if_pitch_greater_than": 30,
        # initial pose — Dodo is ~0.55m tall, spawn at ~0.45m
        "base_init_pos":  [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }

    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.35,
        "reward_scales": {
            # 1. главная задача
            "tracking_lin_vel":   5.0,
            "tracking_ang_vel":   1.0,
            "heading":            0.5,
            # 2. выживание и поза
            "alive":              0.5,
            "upright":            1.0,
            "fall_penalty":      -5.0,
            "base_height":       -40.0,
            "lin_vel_z":         -1.0,
            "similar_to_default":-0.5,
            # 3. эффективность
            "mechanical_power":  -3e-5,
            "torques":           -0.0002,
            "dof_acc":           -1e-7,
            "action_rate":       -0.01,
            # 4. контакт с землёй
            "foot_slip":         -0.05,
            "feet_impact":       -3e-6,
            "feet_orientation":  -0.8,
            "unallowed_contacts":-1.0,
            # 5. походка
            "gait_alternation":   1.0,
            "gait_symmetry":      0.3,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.3, 1.0],  # include standing (0) and backward
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range":   [-0.5, 0.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dodo-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend (useful for macOS testing)")
    parser.add_argument("--viewer", action="store_true", help="Show interactive viewer")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint to resume from")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in the run dir")
    parser.add_argument("--record", action="store_true",
                        help="Record training videos")
    parser.add_argument("--record-every", type=int, default=50,
                        help="Record a video every N iterations (default: 50)")
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), f"../runs/{args.exp_name}")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)

    # resolve checkpoint path: --checkpoint wins over --resume
    checkpoint = args.checkpoint
    if checkpoint is None and args.resume:
        candidates = sorted(Path(log_dir).glob("model_*.pt"),
                            key=lambda p: int(p.stem.split("_")[1]))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {log_dir}")
        checkpoint = str(candidates[-1])
        print(f"Resuming from {checkpoint}")

    # don't wipe log_dir when resuming
    if checkpoint is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=not args.cpu,
    )

    env = DodoEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.viewer,
        enable_render=args.record,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if checkpoint is not None:
        runner.load(checkpoint)

    # curriculum: enable push perturbations after robot learns to walk
    _curriculum_log = runner.logger.log
    def _curriculum_hook(**kw):
        it = kw["it"]
        if it >= 1000 and env.push_interval == 0:
            env.push_interval = 150
            print(f"[curriculum] iter {it}: push perturbations enabled (interval=150, force=10-50N)")
        _curriculum_log(**kw)
    runner.logger.log = _curriculum_hook

    if args.record:
        video_dir = Path(os.path.join(log_dir, "videos"))
        recorder = VideoRecorder(
            env=env,
            video_dir=video_dir,
            every_n_iter=args.record_every,
        )

        _original_step = env.step

        def _step_hook(actions):
            result = _original_step(actions)
            recorder.on_step()
            return result

        env.step = _step_hook

        _original_log = runner.logger.log

        def _log_hook(**kw):
            it = kw["it"]
            recorder.on_iteration_start(it)
            _original_log(**kw)
            recorder.on_iteration_end(it)

        runner.logger.log = _log_hook

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    if args.viewer:
        while True:
            env.scene.step()


if __name__ == "__main__":
    main()
