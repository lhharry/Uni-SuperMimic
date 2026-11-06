"""Franka Panda constants for the fixed-base SuperMimic port.

Gains are the real robot's joint impedance (panda-py JointPosition, values from
franka_pi05/pi05_left.py); the task MJCF uses position actuators with the same
gains, so q_sp = q0 + xi + alpha * a is the setpoint sent to the robot.
Q0 / ALPHA come from the retargeted references (tools/build_panda_motions.py):
q0 = mid-range of each joint, alpha = max deviation / 4.
"""

from __future__ import annotations

ARMS = {"left": ("left",), "dual": ("left", "right")}

MJCF = {
    "left": "src/unisupermimic/assets/robots/panda/panda_left_task.xml",
    "dual": "src/unisupermimic/assets/robots/panda/panda_left_right_task.xml",
}
ROOT_BODY = "mount"

KP = [360.0, 360.0, 360.0, 360.0, 150.0, 90.0, 30.0]
KD = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0]
TAU_MAX = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]

Q0 = {
    "left": [-0.508, -0.953, 0.025, -2.146, 0.046, 1.744, 0.039],
    "right": [0.493, -0.939, -0.008, -2.138, 0.001, 2.119, -0.039],
}
ALPHA = {
    "left": [0.501, 0.146, 0.220, 0.183, 0.094, 0.127, 0.067],
    "right": [0.498, 0.140, 0.207, 0.179, 0.144, 0.255, 0.096],
}


def joint_names(variant: str) -> list[str]:
    return [f"{arm}/joint{i}" for arm in ARMS[variant] for i in range(1, 8)]


def tracked_bodies(variant: str) -> list[str]:
    """Elbow (link4) and wrist (link7) of every arm."""
    return [f"{arm}/{link}" for arm in ARMS[variant] for link in ("link4", "link7")]


def per_joint(table: dict[str, list[float]] | list[float], variant: str) -> list[float]:
    if isinstance(table, dict):
        return [v for arm in ARMS[variant] for v in table[arm]]
    return [v for _ in ARMS[variant] for v in table]
