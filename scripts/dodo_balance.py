#!/usr/bin/env python3
import argparse
import os
import pickle
import shutil
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
        "kp": [20.0, 20.0, 10.0, 10.0, 20.0, 20.0, 10.0, 10.0],
        "kd": [0.5,  0.5,  0.3,  0.3,  0.5,  0.5,  0.3,  0.3],
        "termination_if_roll_greater_than":  30,
        "termination_if_pitch_greater_than": 30,
        "base_init_pos":  [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 10.0,
        "resampling_time_s": 10.0,  # no resampling — always zero command
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
        "base_height_target": 0.30,
        "reward_scales": {
            "upright":             5.0,    # keep body vertical
            "feet_orientation":   -5.0,    # keep feet parallel to ground
            "alive":               2.0,    # reward per step for not falling
            "episode_success":    50.0,    # bonus for surviving full episode
            "fall_penalty":       -10.0,   # penalty on termination
            "unallowed_contacts": -1.0,    # non-foot link touches ground
            "lin_vel_z":          -1.0,    # penalize vertical body movement
            "base_ang_vel":       -0.5,    # penalize body rotation
            "action_rate":        -0.05,   # penalize action changes (10x stronger)
            "dof_acc":            -1e-6,   # penalize joint accelerations (4x stronger)
            "torques":            -2e-4,   # penalize motor effort (2x stronger)
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.0],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range":   [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dodo-balance")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), f"../runs/{args.exp_name}")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)

    # don't wipe log_dir if resuming from checkpoint
    if args.checkpoint is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning",
            seed=args.seed, performance_mode=not args.cpu)

    env = DodoEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.checkpoint is not None:
        runner.load(args.checkpoint)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    if args.viewer:
        while True:
            env.scene.step()


if __name__ == "__main__":
    main()
