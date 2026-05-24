#!/usr/bin/env python3
"""Preview the robot's default standing pose with interactive joint control.

Controls:
  ↑↓   — select joint
  ←→   — adjust angle ±0.01 rad
  r    — reset to default
  p    — print current angles
  q / Ctrl+C — quit
"""
import json
import os
import sys
import threading
import termios
import tty

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import genesis as gs

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../assets/robots/dodo")

# 4 logical joints — applied symmetrically to both legs
LOGICAL_JOINTS = ["hip", "upper_leg", "lower_leg", "foot"]

JOINT_LIMITS = {
    "hip":       (-0.35, 0.35),
    "upper_leg": (-1.57, 1.57),
    "lower_leg": (-3.14, 1.40),
    "foot":      (-1.05, 1.57),
}

DEFAULT_ANGLES = {
    "hip":       0.0,
    "upper_leg": 0.0,
    "lower_leg": -0.4,
    "foot":      1.0,
}

# order Genesis expects for all 8 DOFs
ALL_JOINT_NAMES = [
    "hip_right", "upper_leg_right", "lower_leg_right", "foot_right",
    "hip_left",  "upper_leg_left",  "lower_leg_left",  "foot_left",
]

BASE_INIT_POS  = [0.0, 0.0, 0.42]
BASE_INIT_QUAT = [1.0, 0.0, 0.0, 0.0]

POSE_FILE = os.path.join(os.path.dirname(__file__), "preview_pose.json")


def load_angles():
    if os.path.exists(POSE_FILE):
        try:
            data = json.load(open(POSE_FILE))
            return [data.get(name, DEFAULT_ANGLES[name]) for name in LOGICAL_JOINTS]
        except Exception:
            pass
    return list(DEFAULT_ANGLES.values())


def save_angles():
    data = {name: current_angles[i] for i, name in enumerate(LOGICAL_JOINTS)}
    with open(POSE_FILE, "w") as f:
        json.dump(data, f, indent=2)


current_angles = load_angles()  # 4 values, mirrored to both legs
selected_joint = 0
dirty = threading.Event()       # redraw only
reset_dirty = threading.Event() # redraw + reset robot
reset_dirty.set()


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def all_dof_angles():
    """Expand 4 logical angles to 8 DOF values (right then left)."""
    return current_angles + current_angles


def println(s=""):
    sys.stdout.write(s + "\r\n")
    sys.stdout.flush()


def print_state():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    println("=== Dodo Joint Preview (both legs mirrored) ===")
    println("↑↓: select joint | ←→: ±0.01 rad | r: reset | p: print | q: quit")
    println()
    for i, name in enumerate(LOGICAL_JOINTS):
        lo, hi = JOINT_LIMITS[name]
        marker = ">>>" if i == selected_joint else "   "
        bar_len = 24
        frac = (current_angles[i] - lo) / (hi - lo)
        filled = int(frac * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        println(f"{marker} {name:12s}  {current_angles[i]:+.3f} rad  [{bar}]  ({lo:.2f}..{hi:.2f})")
    println()


def read_keys():
    global selected_joint
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('q', '\x03'):
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                sys.stdout.write("\r\n")
                os._exit(0)
            elif ch == 'r':
                for i, name in enumerate(LOGICAL_JOINTS):
                    current_angles[i] = DEFAULT_ANGLES[name]
                save_angles()
                reset_dirty.set()
            elif ch == 'p':
                println()
                println("Current angles (both legs):")
                for name, val in zip(LOGICAL_JOINTS, current_angles):
                    println(f'    "{name}_right/left": {val:.4f},')
            elif ch == '\x1b':
                seq = sys.stdin.read(2)
                name = LOGICAL_JOINTS[selected_joint]
                lo, hi = JOINT_LIMITS[name]
                if seq == '[A':    # up — previous joint
                    selected_joint = (selected_joint - 1) % len(LOGICAL_JOINTS)
                    dirty.set()
                elif seq == '[B':  # down — next joint
                    selected_joint = (selected_joint + 1) % len(LOGICAL_JOINTS)
                    dirty.set()
                elif seq == '[C':  # right — increase
                    current_angles[selected_joint] = clamp(current_angles[selected_joint] + 0.01, lo, hi)
                    save_angles()
                    reset_dirty.set()
                elif seq == '[D':  # left — decrease
                    current_angles[selected_joint] = clamp(current_angles[selected_joint] - 0.01, lo, hi)
                    save_angles()
                    reset_dirty.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    gs.init(backend=gs.cpu, precision="32", logging_level="error")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.2),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=os.path.join(ASSETS_DIR, "urdf/dodo_daimao.urdf"),
            pos=BASE_INIT_POS,
            quat=BASE_INIT_QUAT,
        )
    )

    scene.build(n_envs=1)

    dof_indices = torch.tensor(
        [robot.get_joint(name).dof_start for name in ALL_JOINT_NAMES],
        dtype=gs.tc_int,
        device=gs.device,
    )

    robot.set_dofs_kp([20.0, 20.0, 10.0, 10.0, 20.0, 20.0, 10.0, 10.0], dof_indices)
    robot.set_dofs_kv([0.5,  0.5,  0.3,  0.3,  0.5,  0.5,  0.3,  0.3],  dof_indices)

    threading.Thread(target=read_keys, daemon=True).start()

    import math

    def compute_spawn_quat():
        # All joints rotate around Y (pitch). Total foot pitch relative to body
        # = upper_leg + lower_leg + foot. Tilt body by the negative to cancel.
        idx = {name: i for i, name in enumerate(LOGICAL_JOINTS)}
        pitch = -(current_angles[idx["upper_leg"]] +
                  current_angles[idx["lower_leg"]] +
                  current_angles[idx["foot"]])
        cy, sy = math.cos(pitch / 2), math.sin(pitch / 2)
        # quaternion for rotation around Y: (w, x, y, z)
        return [cy, 0.0, sy, 0.0]

    base_pos = torch.tensor([BASE_INIT_POS], dtype=gs.tc_float, device=gs.device)

    def reset_robot():
        dof_pos = torch.tensor([all_dof_angles()], dtype=gs.tc_float, device=gs.device)
        base_quat = torch.tensor([compute_spawn_quat()], dtype=gs.tc_float, device=gs.device)
        robot.set_pos(base_pos)
        robot.set_quat(base_quat)
        robot.set_dofs_position(dof_pos, dof_indices)
        robot.zero_all_dofs_velocity()
        robot.control_dofs_position(dof_pos, dof_indices)

    # masses from URDF
    LINK_MASSES = {
        "body":            3.3120,
        "hip_right":       0.4286,
        "upper_leg_right": 0.1400,
        "lower_leg_right": 0.0730,
        "foot_right":      0.0358,
        "hip_left":        0.4286,
        "upper_leg_left":  0.1400,
        "lower_leg_left":  0.0730,
        "foot_left":       0.0358,
    }
    total_mass = sum(LINK_MASSES.values())
    link_map = {link.name: link for link in robot.links}

    def compute_com_and_support():
        # centre of mass (x only — forward/backward balance)
        com_x = 0.0
        for name, mass in LINK_MASSES.items():
            if name in link_map:
                pos = link_map[name].get_pos()[0]
                com_x += mass * pos[0].item()
        com_x /= total_mass

        # support centre = midpoint between feet in X
        foot_r_x = link_map["foot_right"].get_pos()[0, 0].item()
        foot_l_x = link_map["foot_left"].get_pos()[0, 0].item()
        support_x = (foot_r_x + foot_l_x) / 2.0

        return com_x, support_x

    def print_balance():
        com_x, support_x = compute_com_and_support()
        offset = com_x - support_x
        bar_len = 21
        center = bar_len // 2
        tick = int(offset * 40)  # scale: 1m = 40 chars
        tick = max(-center, min(center, tick))
        bar = [" "] * bar_len
        bar[center] = "|"
        bar[center + tick] = "█"
        bar_str = "[" + "".join(bar) + "]"
        status = "OK" if abs(offset) < 0.02 else ("FORWARD" if offset > 0 else "BACK")
        sys.stdout.write(f"\r  CoM offset: {offset:+.4f} m  {bar_str}  {status}   ")
        sys.stdout.flush()

    print_state()
    while True:
        if reset_dirty.is_set():
            reset_dirty.clear()
            dirty.clear()
            reset_robot()
            print_state()
        elif dirty.is_set():
            dirty.clear()
            print_state()
        scene.step()
        print_balance()


if __name__ == "__main__":
    main()
