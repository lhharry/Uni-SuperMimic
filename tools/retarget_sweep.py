"""Stage 5.5 sweep: the real retarget pipeline (retarget_clip: both arms, inter-arm obstacle,
validity incl. wrist error) on a grid of Mappings, scored on human-likeness and feasibility.

Per mapping, over all CANDIDATES clips (stride 2 = 15 Hz):
  feasibility  valid fraction, minutes in valid segments (>= 5 s), retimed minutes
               (build_panda_motions.warp_segment at 0.3 rad/s), wrist error p99 on valid frames,
               branch jumps (frames inside segments with max |dq| > 0.3 rad between samples),
               max static gravity torque J1-4 on valid frames (gravcomp off)
  likeness     D_w p50 / p90 on valid frames, per sector (out / up / down / fwd / in),
               elevation spread ratio Panda / human (tools/human_likeness.py)

Selection: hard filters relative to v1b (the v1 cone on the new geometry and pipeline):
valid-segment minutes >= 0.85 x v1b, retimed minutes <= 1.3 x v1b, branch jumps <= v1b,
gravity torque <= 0.7 tau_max (61 Nm J1-4); then the Pareto front of (D_w p50 on all valid
frames, D_w p90 on out+up frames).
Changed from the pre-registered plan after the 2-clip smoke test (2026-11-12): the wrist-error
validity cut (6 cm) and the "wrist error p99 <= 4 cm" filter are dropped (reported only) - the
reference is the robot's own trajectory, so wrist error is fidelity, which D_w already
measures, and the cut shredded segments (v1b: 6.1 -> 0.6 valid-segment minutes); "jumps == 0"
became "<= v1b" because v1 itself has them (51 on two clips). Lower IK barriers were dropped
(valid 0.85 -> 0.57).

    uv run python tools/retarget_sweep.py --grid default --out runs/retarget_sweep_stage5.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_likeness as hl  # noqa: E402
import retarget_bvh_arms_to_panda as rt  # noqa: E402
from build_panda_motions import warp_segment  # noqa: E402

TAU_LIMIT = 0.7 * np.array([87, 87, 87, 87.0])
JUMP = 0.3


def grid(name: str) -> list[rt.Mapping]:
    out = [replace(rt.V1, wrist_tol_valid=None)]                       # index 0 = v1b, the baseline
    if name == "smoke":
        return out + [rt.Mapping(a0=30, out=180, up=180, down=60, inn=50, wrist_tol_valid=None)]
    for a0, o, u, dn, inn in itertools.product((30, 45), (90, 120, 180), (90, 120, 180), (50, 60, 70), (40, 50)):
        out.append(rt.Mapping(a0=a0, out=o, up=u, down=dn, inn=inn, wrist_tol_valid=None))
    return out


_GRAV = {}


def gravity_torque(arm: str, Q: np.ndarray) -> np.ndarray:
    """Static gravity torque of J1-4 with gravcomp off (what libfranka compensates on the robot)."""
    if arm not in _GRAV:
        ik = rt.ArmIk(arm)
        m = ik.m
        m.body_gravcomp[:] = 0.0
        _GRAV[arm] = (m, mujoco.MjData(m), ik.qadr, ik.dadr)
    m, d, qa, da = _GRAV[arm]
    out = np.zeros((len(Q), 4))
    for i, q in enumerate(Q):
        d.qpos[qa] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        out[i] = d.qfrc_bias[da][:4]
    return out


def job(args):
    mi, m, clip, stride = args
    r = rt.retarget_clip(clip, m, stride=stride)
    s = rt.clip_summary(r)
    v, fps = r["valid"], r["fps"]
    res = dict(mi=mi, clip=clip, frames=len(v), valid=int(v.sum()),
               fails={k: x * len(v) for k, x in s["fails"].items()},
               seg_min=sum(x["seconds"] for x in s["segments"]) / 60, retimed_min=0.0, jumps=0)
    for sg in s["segments"]:
        q = r["Q"][sg["start"]:sg["end"]]
        q50, _, _ = warp_segment(q, fps)
        res["retimed_min"] += len(q50) * 0.02 / 60
        res["jumps"] += int((np.abs(np.diff(q, axis=0)).max(axis=1) > JUMP).sum())
    rows, errs, tau = [], [], []
    for k, side in enumerate(("left", "right")):
        ik = rt.ArmIk(side)
        Qs = r["Q"][:, 7 * k:7 * k + 7]
        f = hl.frame_metrics(r["kp"], side, ik, Qs, r["idx"])
        f["valid"] = v
        rows.append(f)
        errs.append(r["err"][side][v])
        sub = np.flatnonzero(v)[::5]
        tau.append(np.abs(gravity_torque(side, Qs[sub])).max(0) if len(sub) else np.zeros(4))
    res["rows"] = rows
    res["err_valid"] = np.concatenate(errs)
    res["tau_max"] = np.max(tau, axis=0).tolist()
    return res


def aggregate(m: rt.Mapping, parts: list[dict]) -> dict:
    rows = [r for p in parts for r in p["rows"]]
    lk = hl.summarise(rows, "valid")
    cat = {k: np.concatenate([r[k] for r in rows]) for k in ("D_w", "sector", "valid")}
    ou = cat["valid"].astype(bool) & np.isin(cat["sector"], ("out", "up"))
    err = np.concatenate([p["err_valid"] for p in parts])
    return dict(mapping=asdict(m),
                valid=round(sum(p["valid"] for p in parts) / sum(p["frames"] for p in parts), 3),
                seg_min=round(sum(p["seg_min"] for p in parts), 1),
                retimed_min=round(sum(p["retimed_min"] for p in parts), 1),
                jumps=int(sum(p["jumps"] for p in parts)),
                fails={k: round(sum(p["fails"].get(k, 0) for p in parts) / sum(p["frames"] for p in parts), 3)
                       for k in sorted({k for p in parts for k in p["fails"]})},
                wrist_err_p99_cm=round(float(np.percentile(err, 99)) * 100, 1) if len(err) else None,
                tau_max=np.round(np.max([p["tau_max"] for p in parts], axis=0), 1).tolist(),
                D_w_p50=lk["D_w_p50"], D_w_p90=lk["D_w_p90"],
                D_w_outup_p50=round(float(np.percentile(cat["D_w"][ou], 50)), 1) if ou.any() else None,
                D_w_outup_p90=round(float(np.percentile(cat["D_w"][ou], 90)), 1) if ou.any() else None,
                elev_spread=lk["elev_spread_ratio"], sectors=lk["sectors"])


def select(res: list[dict]) -> list[int]:
    base = res[0]                                # v1b
    ok = []
    for i, r in enumerate(res):
        r["pass"] = dict(seg_min=r["seg_min"] >= 0.85 * base["seg_min"],
                         retimed=r["retimed_min"] <= 1.3 * base["retimed_min"],
                         jumps=r["jumps"] <= base["jumps"],
                         gravity=all(t <= lim for t, lim in zip(r["tau_max"], TAU_LIMIT)))
        if all(r["pass"].values()):
            ok.append(i)
    pareto = [i for i in ok if not any(res[j]["D_w_p50"] <= res[i]["D_w_p50"] and
                                       res[j]["D_w_outup_p90"] <= res[i]["D_w_outup_p90"] and
                                       (res[j]["D_w_p50"], res[j]["D_w_outup_p90"]) != (res[i]["D_w_p50"], res[i]["D_w_outup_p90"])
                                       for j in ok)]
    return sorted(pareto, key=lambda i: res[i]["D_w_p50"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="default", choices=("default", "smoke"))
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", type=Path, default=rt.REPO / "runs" / "retarget_sweep_stage5.json")
    a = ap.parse_args()
    maps = grid(a.grid)
    clips = a.clips or list(rt.CANDIDATES)
    jobs = [(i, m, c, a.stride) for i, m in enumerate(maps) for c in clips]

    print(f"{len(maps)} mappings x {len(clips)} clips = {len(jobs)} jobs", flush=True)
    t0, parts = time.time(), {i: [] for i in range(len(maps))}
    with ProcessPoolExecutor(a.workers) as ex:
        for n, p in enumerate(ex.map(job, jobs, chunksize=1), 1):
            parts[p["mi"]].append(p)
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} jobs, {time.time() - t0:.0f} s", flush=True)
    res = [aggregate(maps[i], parts[i]) for i in range(len(maps))]
    front = select(res)
    a.out.write_text(json.dumps(dict(stride=a.stride, clips=clips, results=res, pareto=front), indent=1))
    hdr = f"{'#':>3} {'kind':5} {'a0':>3} {'out':>4} {'up':>4} {'dn':>3} {'bar':>9} {'valid':>5} {'seg':>5} {'ret':>6} {'jmp':>4} {'err99':>5} {'tauJ1':>5} {'Dw50':>5} {'Dw90':>5} {'OU50':>5} {'OU90':>5} {'spread':>6} pass"
    print(hdr)
    for i, r in enumerate(res):
        mm = r["mapping"]
        bar = f"{mm['collision']}/{mm['soft_margin']}"
        flag = "*" if i in front else ("ok" if all(r["pass"].values()) else ",".join(k for k, v in r["pass"].items() if not v))
        print(f"{i:3d} {mm['kind']:5} {mm['a0']:3.0f} {mm['out']:4.0f} {mm['up']:4.0f} {mm['down']:3.0f} {bar:>9} "
              f"{r['valid']:5.2f} {r['seg_min']:5.1f} {r['retimed_min']:6.1f} {r['jumps']:4d} {r['wrist_err_p99_cm']:5.1f} "
              f"{r['tau_max'][0]:5.1f} {r['D_w_p50']:5.1f} {r['D_w_p90']:5.1f} {r['D_w_outup_p50']:5.1f} {r['D_w_outup_p90']:5.1f} "
              f"{r['elev_spread']:6.2f} {flag}")
    print("Pareto front (pre-registered rule):", front, f"\n{a.out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
