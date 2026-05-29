#!/usr/bin/env python3
"""
Load a checkpoint and let a gamepad drive the robot in real-time.

Controls (Xbox / DualShock layout):
  Left stick   →  walk (forward/back/strafe)
  Right stick  →  camera orbit (pan/tilt)
  Button B / Circle  →  quit

Usage:
  python scripts/gamepad_play.py --checkpoint runs/dodo-balance/model_500.pt
  python scripts/gamepad_play.py --checkpoint runs/dodo-balance/model_500.pt --max-vel 0.8
"""
import argparse
import numpy as np
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--exp-name", default=None,
                   help="Experiment name (inferred from checkpoint path if omitted)")
    p.add_argument("--max-vel", type=float, default=0.5,
                   help="Max linear velocity magnitude sent as command (default: 0.5 m/s)")
    p.add_argument("--max-ang", type=float, default=1.0,
                   help="Max angular velocity (default: 1.0 rad/s)")
    p.add_argument("--deadzone", type=float, default=0.08,
                   help="Stick deadzone (0-1, default: 0.08)")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _init_pygame():
    import pygame
    pygame.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    if n == 0:
        print("[gamepad] No joystick detected — using keyboard fallback (WASD + Q/E to turn, Esc to quit)")
        return None
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"[gamepad] Using: {joy.get_name()}")
    # create a tiny hidden window so pygame event loop works
    pygame.display.set_mode((1, 1), pygame.NOFRAME)
    return joy


def _read_gamepad(joy, deadzone, max_vel, max_ang):
    """Return (vx, vy, wz, cam_yaw, cam_pitch, quit_requested).
    Left stick:  axis0=X (strafe), axis1=Y (forward)
    Right stick: axis2=X (cam yaw), axis3=Y (cam pitch)
    LB (btn9)=turn left, RB (btn10)=turn right, B (btn1)=quit
    """
    import pygame
    pygame.event.pump()  # flush OS events so get_axis/get_button stay fresh

    def ax(idx):
        v = joy.get_axis(idx)
        return v if abs(v) > deadzone else 0.0

    rt        = (joy.get_axis(5) + 1.0) / 2.0  # -1..+1 → 0..1
    boost     = 1.0 + rt * 2.0                 # 1x..3x speed
    vx        = -ax(1) * max_vel * boost
    vy        = -ax(0) * max_vel * boost
    wz        = (joy.get_button(9) - joy.get_button(10)) * max_ang
    cam_yaw   =  ax(2)
    cam_pitch =  ax(3)
    reset_req = bool(joy.get_button(1))   # B = respawn
    quit_req  = bool(joy.get_button(7))   # Start/Menu = quit
    return vx, vy, wz, cam_yaw, cam_pitch, reset_req, quit_req


def _read_keyboard(deadzone, max_vel, max_ang):
    """Fallback: WASD steering via pygame keyboard."""
    import pygame
    quit_req = False
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            quit_req = True
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            quit_req = True

    keys = pygame.key.get_pressed()
    vx = (keys[pygame.K_w] - keys[pygame.K_s]) * max_vel
    vy = (keys[pygame.K_a] - keys[pygame.K_d]) * max_vel
    wz = (keys[pygame.K_q] - keys[pygame.K_e]) * max_ang
    return float(vx), float(vy), float(wz), quit_req


def main():
    args = parse_args()

    try:
        import pygame  # noqa: F401
    except ImportError:
        print("pygame not installed. Run:  pip install pygame")
        sys.exit(1)

    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    # infer exp_name for log_dir (used by OnPolicyRunner for logging)
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = os.path.basename(checkpoint_dir)
    log_dir = checkpoint_dir

    # load cfgs from the same folder as the checkpoint
    cfgs_path = os.path.join(checkpoint_dir, "cfgs.pkl")
    if os.path.exists(cfgs_path):
        with open(cfgs_path, "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
        print(f"[gamepad] Loaded cfgs from {cfgs_path}")
    else:
        from dodo_balance import get_cfgs, get_train_cfg
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(exp_name)
        print("[gamepad] cfgs.pkl not found — using defaults from dodo_balance")

    # allow full velocity range so the gamepad can reach any speed
    command_cfg["lin_vel_x_range"] = [-args.max_vel, args.max_vel]
    command_cfg["lin_vel_y_range"] = [-args.max_vel, args.max_vel]
    command_cfg["ang_vel_range"]   = [-args.max_ang, args.max_ang]

    # no episode resets during play — let it run until we quit
    env_cfg["episode_length_s"] = 3600.0
    env_cfg["resampling_time_s"] = 3600.0

    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    from src.envs.dodo_env import DodoEnv

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    env = DodoEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        enable_render=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.checkpoint, map_location=str(gs.device))
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[gamepad] Checkpoint loaded: {args.checkpoint}")

    # disable the env's built-in follow_entity so we can control camera manually
    env.scene.viewer._followed_entity = None

    joy = _init_pygame()
    read_input = (lambda: _read_gamepad(joy, args.deadzone, args.max_vel, args.max_ang)) \
                 if joy else (lambda: _read_keyboard(args.deadzone, args.max_vel, args.max_ang))
    import math

    def _get_yaw():
        q = env.base_quat[0].cpu().numpy()
        return math.atan2(
            2.0 * (q[0]*q[3] + q[1]*q[2]),
            1.0 - 2.0 * (q[2]**2 + q[3]**2)
        )

    obs = env.reset()
    cam_azimuth       = _get_yaw() + math.pi
    cam_elevation     = 0.28
    cam_elevation_min = -0.05
    cam_elevation_max = 1.1
    cam_radius        = 2.5
    cam_yaw_speed     = 2.0

    print("[gamepad] Running — left stick=walk, right stick=camera.  B / Esc = quit")

    while True:
        result = read_input()
        if len(result) == 7:
            vx, vy, wz, cam_dyaw, cam_dpitch, reset_req, quit_req = result
        else:
            vx, vy, wz, quit_req = result
            cam_dyaw = cam_dpitch = reset_req = 0.0

        if quit_req:
            break

        if reset_req:
            obs = env.reset()
            continue

        # convert stick input from camera frame to robot body frame
        # so "forward" on stick = camera forward, not robot forward
        robot_yaw = _get_yaw()
        cam_az = cam_azimuth - math.pi
        world_vx = vx * math.cos(cam_az) - vy * math.sin(cam_az)
        world_vy = vx * math.sin(cam_az) + vy * math.cos(cam_az)
        body_vx =  world_vx * math.cos(robot_yaw) + world_vy * math.sin(robot_yaw)
        body_vy = -world_vx * math.sin(robot_yaw) + world_vy * math.cos(robot_yaw)

        env._world_cmd = (world_vx, world_vy)  # for arrow rendering — no yaw jitter
        env.commands[:, 0] = body_vx
        env.commands[:, 1] = body_vy
        env.commands[:, 2] = wz

        with torch.no_grad():
            actions = policy(obs)

        obs, _rew, reset, _info = env.step(actions)

        # --- third-person camera (after step so base_pos/quat are fresh) ---
        robot_pos = env.base_pos[0].cpu().numpy()

        CAM_DEADZONE = 0.2
        if abs(cam_dyaw) < CAM_DEADZONE:
            cam_dyaw = 0.0
        if abs(cam_dpitch) < CAM_DEADZONE:
            cam_dpitch = 0.0
        cam_azimuth += cam_dyaw * cam_yaw_speed * env.dt
        cam_elevation -= cam_dpitch * 1.2 * env.dt
        cam_elevation = max(cam_elevation_min, min(cam_elevation_max, cam_elevation))

        lookat  = np.array([robot_pos[0], robot_pos[1], robot_pos[2] + 0.35])
        cam_pos = lookat + cam_radius * np.array([
            math.cos(cam_elevation) * math.cos(cam_azimuth),
            math.cos(cam_elevation) * math.sin(cam_azimuth),
            math.sin(cam_elevation),
        ])

        # write directly to both trackball poses so it never self-rotates
        # (set_camera_pose only sets _n_pose; _pose stays stale and causes drift)
        v = env.scene.viewer
        if hasattr(v, '_pyrender_viewer'):
            import numpy.linalg as LA
            fwd   = lookat - cam_pos;  fwd /= LA.norm(fwd)
            right = np.cross(fwd, [0., 0., 1.]);  right /= LA.norm(right)
            up    = np.cross(right, fwd)
            R = np.eye(4)
            R[:3, 0] = right;  R[:3, 1] = up;  R[:3, 2] = -fwd;  R[:3, 3] = cam_pos
            tb = v._pyrender_viewer._trackball
            tb._pose   = R.copy()
            tb._n_pose = R.copy()

        # reset only on fall, never on timeout
        if reset.any():
            timed_out = _info.get("time_outs", torch.zeros(1, device=gs.device)).bool()
            fell = reset & ~timed_out
            if fell.any():
                print("[gamepad] Robot fell — resetting")
                obs = env.reset()

    print("[gamepad] Done.")
    import pygame
    pygame.quit()


if __name__ == "__main__":
    main()
