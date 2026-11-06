"""Build the wall-mounted dual-Panda MJCF for the SuperMimic port.

World frame (human-like, the robot "faces" out of the mounting panel):
  +x  out of the panel (toward the operator)
  +y  the robot's left (the operator's right)
  +z  up
  origin: midpoint between the two J1 axes, on the base mounting surface.

Geometry comes from ~/harry/franka_mimic/config/stage0_geometry.yaml (tape
measurements + read_state.py gravity fits, 2026-11-06):
  - J1 axes 0.59 m apart, 0.64 m above the table, mounting surface 0.01 m
    in front of the panel.
  - panda_left (192.168.3.100, Franka Hand) at +y, gravity along its base +Y.
  - panda_right (192.168.4.100, no hand) at -y, gravity along its base -Y,
    i.e. rotated 180 deg about the panel normal (mirror mount). Both base X
    axes point toward the other arm.
  - table top 0.64 m below J1, front edge 0.86 m from the panel, side edges
    0.60 m outboard of each base.

Fingers are welded (no finger joints), so the model has exactly 7 hinge joints
per arm and one torque motor per joint. All arm bodies use gravcomp=1 to match
libfranka, which compensates gravity internally.

Keep-out volumes (side walls, the unused head unit above the panel centre) are
non-colliding geoms in group 5; retargeting and safety checks measure distance
to them.

    uv run python tools/build_panda_mjcf.py            # writes left-only and dual
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MENAGERIE = REPO / "data" / "mujoco_menagerie" / "franka_emika_panda"
OUT_DIR = REPO / "src" / "unisupermimic" / "assets" / "robots" / "panda"

# --- workcell geometry (metres) -------------------------------------------
BASE_SPACING = 0.59
J1_ABOVE_TABLE = 0.64
MOUNT_OFFSET = 0.01          # base mounting surface in front of the panel
TABLE_DEPTH = 0.86           # front edge measured from the panel
TABLE_SIDE_OUTBOARD = 0.60   # table side edge outboard of each base
PANEL_TOP = 0.30             # above J1 (estimated from photos)
HEAD_KEEPOUT = dict(y=0.15, depth=0.30, z0=PANEL_TOP)  # unused head unit

TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
# Real-robot joint impedance (panda-py JointPosition, franka_pi05/pi05_left.py); the task
# models use position actuators with exactly these gains: tau = K (q_sp - q) - D qd.
KP = np.array([360., 360., 360., 360., 150., 90., 30.])
KD = np.array([50., 50., 50., 20., 20., 20., 10.])
# Neutral pose q0 (mid-range of the retargeted references, tools/build_panda_motions.py).
Q0 = {"left": [-0.508, -0.953, 0.025, -2.146, 0.046, 1.744, 0.039],
      "right": [0.493, -0.939, -0.008, -2.138, 0.001, 2.119, -0.039]}

ARMS = {
    # name: (y offset, base X axis, base Y axis) in world; base Z = +x (out of panel)
    "left": (+BASE_SPACING / 2, (0, -1, 0), (0, 0, -1)),
    "right": (-BASE_SPACING / 2, (0, 1, 0), (0, 0, 1)),
}
HAND = {"left": True, "right": False}


def base_quat(x_axis, y_axis) -> np.ndarray:
    x, y = np.asarray(x_axis, float), np.asarray(y_axis, float)
    R = np.stack([x, y, np.cross(x, y)], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def load_arm(hand: bool) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / ("panda.xml" if hand else "panda_nohand.xml")))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    if hand:
        for t in list(spec.tendons):
            spec.delete(t)
        for e in list(spec.equalities):
            spec.delete(e)
        for j in ("finger_joint1", "finger_joint2"):
            spec.delete(spec.joint(j))
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    return spec


def build(arms: list[str], task: bool = False) -> mujoco.MjSpec:
    """task=False: torque motors (analysis tools). task=True: SuperMimic task model with a
    static 'mount' root body, real-robot PD position actuators and a 'home' keyframe at q0."""
    w = mujoco.MjSpec()
    w.modelname = "panda_" + "_".join(arms) + ("_task" if task else "")
    w.compiler.degree = False
    w.meshdir = "assets"
    w.option.timestep = 0.002
    w.option.impratio = 10
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.stat.center = [0.4, 0, -0.2]
    w.stat.extent = 1.0

    wb = w.worldbody
    wb.add_light(pos=[1.5, 0, 1.5], dir=[-1, 0, -1], diffuse=[0.8, 0.8, 0.8])
    wb.add_camera(name="front", pos=[2.2, 0, 0.3], xyaxes=[0, 1, 0, -0.2, 0, 1])

    # panel (a static body so contacts with link0/link1 can be excluded)
    ymax = BASE_SPACING / 2 + TABLE_SIDE_OUTBOARD
    panel = wb.add_body(name="panel")
    panel.add_geom(name="panel", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.01, ymax + 0.3, (PANEL_TOP + J1_ABOVE_TABLE) / 2],
                   pos=[-MOUNT_OFFSET - 0.01, 0, (PANEL_TOP - J1_ABOVE_TABLE) / 2],
                   rgba=[0.6, 0.6, 0.62, 1])
    table = wb.add_body(name="table")
    table.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[TABLE_DEPTH / 2, ymax, 0.02],
                   pos=[TABLE_DEPTH / 2 - MOUNT_OFFSET, 0, -J1_ABOVE_TABLE - 0.02],
                   rgba=[0.3, 0.3, 0.32, 1])
    keep = wb.add_body(name="keepout")
    red = [1, 0.2, 0.2, 0.15]
    for side, s in (("keepout_side_left", 1), ("keepout_side_right", -1)):
        keep.add_geom(name=side, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.01, 0.8],
                      pos=[0.5, s * (ymax + 0.01), 0], contype=0, conaffinity=0, group=5, rgba=red)
    keep.add_geom(name="keepout_head", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[HEAD_KEEPOUT["depth"] / 2, HEAD_KEEPOUT["y"], 0.4],
                  pos=[HEAD_KEEPOUT["depth"] / 2 - MOUNT_OFFSET, 0, HEAD_KEEPOUT["z0"] + 0.4],
                  contype=0, conaffinity=0, group=5, rgba=red)

    root = wb.add_body(name="mount") if task else wb
    for arm in arms:
        y, xa, ya = ARMS[arm]
        frame = root.add_frame(pos=[0, y, 0], quat=base_quat(xa, ya))
        child = load_arm(HAND[arm])
        frame.attach_body(child.body("link0"), f"{arm}/", "")
        w.add_exclude(bodyname1="panel", bodyname2=f"{arm}/link0")
        w.add_exclude(bodyname1="panel", bodyname2=f"{arm}/link1")

    model = w.compile()
    for arm in arms:
        for i in range(7):
            j = f"{arm}/joint{i + 1}"
            if not task:
                w.add_actuator(name=f"{arm}/motor{i + 1}", target=j,
                               trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrllimited=True,
                               ctrlrange=[-TAU_MAX[i], TAU_MAX[i]])
                continue
            act = w.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = KP[i]
            act.biasprm[:3] = [0.0, -KP[i], -KD[i]]
            act.ctrllimited, act.ctrlrange = True, model.joint(j).range
            act.forcelimited, act.forcerange = True, [-TAU_MAX[i], TAU_MAX[i]]
    if task:
        q0 = np.concatenate([Q0[a] for a in arms])
        w.add_key(name="home", qpos=q0, ctrl=q0)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not (a.out / "assets").exists():
        shutil.copytree(MENAGERIE / "assets", a.out / "assets")
    for arms, task in ((["left"], False), (["left", "right"], False),
                       (["left"], True), (["left", "right"], True)):
        spec = build(arms, task)
        model = spec.compile()
        path = a.out / f"{spec.modelname}.xml"
        xml = spec.to_xml()
        path.write_text(xml)
        mujoco.MjModel.from_xml_path(str(path))  # round-trip check
        print(f"{path.relative_to(REPO)}: nq={model.nq} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")


if __name__ == "__main__":
    main()
