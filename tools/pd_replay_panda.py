"""PD replay gate for the dual-Panda references (plan step 2.3), also the baseline for stage 3.

Each reference segment is tracked in MuJoCo with the real robot's joint
impedance (panda-py JointPosition gains from franka_pi05/pi05_left.py):
    tau = K (q_ref - q) + D (dq_d - dq),  clipped to tau_max
with gravity compensated by the model (gravcomp), reference targets held
between 50 Hz control ticks, physics at 500 Hz. Two variants:
    dq_d = 0       (what the real-robot script will send)
    dq_d = qd_ref  (feed-forward velocity, for comparison)

Reported per segment: joint RMSE / max error, wrist (link6) position error
per arm, torque saturation, contacts, min clearance to panel/table/keep-outs
and between the arms.

    uv run python tools/pd_replay_panda.py            # all segments, both variants
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panda_arm_ik import ArmIk, inter_arm_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "src" / "unisupermimic" / "assets" / "motions" / "panda" / "lafan1_panda_dual.pt"
K = np.tile([360., 360., 360., 360., 150., 90., 30.], 2)
D = np.tile([50., 50., 50., 20., 20., 20., 10.], 2)
TAU_MAX = np.tile([87., 87., 87., 87., 12., 12., 12.], 2)
CONTROL_DT = 0.02
CHECK_EVERY = 5     # clearance check every 5 control ticks (10 Hz)


def replay(args):
    name, ref, ff = args
    J = 14
    q_ref, qd_ref = ref[:, 7:7 + J], ref[:, 13 + J:13 + 2 * J]
    L, R = ArmIk("left"), ArmIk("right")
    m, d = L.m, L.d
    substeps = int(round(CONTROL_DT / m.opt.timestep))
    d.qpos[:] = q_ref[0]
    d.qvel[:] = qd_ref[0] if ff else 0
    mujoco.mj_forward(m, d)
    wl, wr = m.body("left/link6").id, m.body("right/link6").id
    arm_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith(("left/", "right/"))}
    q_err, w_err_l, w_err_r, sat, contacts, clr, inter = [], [], [], 0, 0, [], []
    dref = mujoco.MjData(m)
    for t in range(len(q_ref)):
        # reference wrist positions
        dref.qpos[:] = q_ref[t]
        mujoco.mj_kinematics(m, dref)
        for _ in range(substeps):
            tau = K * (q_ref[t] - d.qpos) + D * ((qd_ref[t] if ff else 0) - d.qvel)
            sat += int(np.any(np.abs(tau) > TAU_MAX))
            d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
            mujoco.mj_step(m, d)
            for c in d.contact[:d.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if (b1 in arm_bodies or b2 in arm_bodies) and c.dist < 0:
                    contacts += 1
        q_err.append(d.qpos - q_ref[t])
        w_err_l.append(np.linalg.norm(d.xpos[wl] - dref.xpos[wl]))
        w_err_r.append(np.linalg.norm(d.xpos[wr] - dref.xpos[wr]))
        if t % CHECK_EVERY == 0:
            for ik in (L, R):
                ik.d.qpos[:] = d.qpos
                mujoco.mj_kinematics(ik.m, ik.d)
            clr.append(min(L.env_clearance()[0], R.env_clearance()[0]))
            inter.append(inter_arm_distance(L, R))
    q_err = np.array(q_err)
    n_steps = len(q_ref) * substeps
    return name, dict(
        seconds=len(q_ref) * CONTROL_DT,
        joint_rmse=float(np.sqrt(np.mean(q_err ** 2))),
        joint_max_err=float(np.abs(q_err).max()),
        wrist_rmse_mm_left=float(1000 * np.sqrt(np.mean(np.square(w_err_l)))),
        wrist_rmse_mm_right=float(1000 * np.sqrt(np.mean(np.square(w_err_r)))),
        wrist_max_mm=float(1000 * max(max(w_err_l), max(w_err_r))),
        sat_frac=sat / n_steps, contacts=contacts,
        min_clearance=float(min(clr)), min_inter_arm=float(min(inter)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="only the first N segments")
    a = ap.parse_args()
    pack = torch.load(PACK)
    ref = pack["ref"].numpy().astype(np.float64)
    segs = [(n, ref[s:s + ln]) for n, s, ln in zip(pack["names"], pack["starts"], pack["lengths"])]
    if a.limit:
        segs = segs[: a.limit]
    out = {}
    for ff in (False, True):
        tag = "ff" if ff else "dq0"
        jobs = [(n, r, ff) for n, r in segs]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = dict(ex.map(replay, jobs, chunksize=1))
        out[tag] = res
        v = list(res.values())
        w = np.array([x["seconds"] for x in v])
        avg = lambda k: float(np.average([x[k] for x in v], weights=w))  # noqa: E731
        print(f"[{tag}] {len(v)} segs  joint_rmse={avg('joint_rmse'):.4f} rad  "
              f"wrist_rmse L/R={avg('wrist_rmse_mm_left'):.1f}/{avg('wrist_rmse_mm_right'):.1f} mm  "
              f"wrist_max={max(x['wrist_max_mm'] for x in v):.1f} mm  "
              f"sat_frac={avg('sat_frac'):.2e}  contacts={sum(x['contacts'] for x in v)}  "
              f"min_clear={min(x['min_clearance'] for x in v):.3f} m  "
              f"min_inter={min(x['min_inter_arm'] for x in v):.3f} m", flush=True)
    dst = REPO / "runs" / "panda_retarget" / "pd_replay.json"
    dst.write_text(json.dumps(out, indent=1))
    print(dst)


if __name__ == "__main__":
    main()
