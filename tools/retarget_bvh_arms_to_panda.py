"""Retarget LAFAN1 BVH arm motion directly to the wall-mounted dual Panda.

Mapping (torso frame of the actor == world frame of the robot, both x forward,
y left, z up). For each arm, with S the Panda J2 centre:
    wrist target  = S + c + k * A (wrist - shoulder)
    elbow hint    = S + L_upper * normalise(A (elbow - shoulder))
    hand dir      = normalise(A (wrist - elbow))
A is a pitch rotation by `pitch_deg` that tilts the actor's "down" toward
"forward"; directions are then squeezed into a cone around +x (compress_cone:
angles beyond a0 map linearly onto [a0, amax]) because the Panda J2 limit
(+-101 deg from the panel normal, minus margin) cannot point along the panel,
and the table sits only 0.64 m below J1. k scales reach; c shifts the workspace. Actor's left arm -> panda_left, right arm -> panda_right.

Modes:
  --sweep   evaluate a grid of (pitch, k, c) on sub-sampled frames of the
            candidate clips and print reachability tables.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bvh_arms import arm_keypoints  # noqa: E402
from panda_arm_ik import ArmIk, IkWeights  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BVH_DIR = REPO / "data" / "lafan1_bvh"
L_UPPER = 0.327
CANDIDATES = (
    [f"dance1_subject{i}" for i in (1, 2, 3)]
    + [f"dance2_subject{i}" for i in (1, 2, 3, 4, 5)]
    + [f"fight1_subject{i}" for i in (2, 3, 5)]
    + [f"fightAndSports1_subject{i}" for i in (1, 4)]
    + ["aiming1_subject1", "aiming1_subject4"]   # aiming2: both hands together on a gun, <25% valid
)
IK_WEIGHTS = IkWeights(elbow=0.02, posture=0.03, step=0.2)   # wrist-led; smooth through J2~0
WRIST_TOL = 0.04
CLEARANCE_MIN = 0.15


def pitch_matrix(deg: float) -> np.ndarray:
    t = np.deg2rad(-deg)   # rotate "down" toward +x
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def compress_cone(v: np.ndarray, a0: float, amax: float) -> np.ndarray:
    """Squeeze directions toward +x: angle a from +x maps to a for a <= a0, and
    linearly [a0, 180] -> [a0, amax] beyond. Keeps lengths. Degrees."""
    if a0 >= 180:
        return v
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    u = v / n
    a = np.arccos(np.clip(u[:, 0], -1, 1))
    a0r, amr = np.deg2rad(a0), np.deg2rad(amax)
    a_new = np.where(a <= a0r, a, a0r + (a - a0r) * (amr - a0r) / (np.pi - a0r))
    perp = u.copy()
    perp[:, 0] = 0
    pn = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = np.where(pn > 1e-6, perp / np.maximum(pn, 1e-9), np.array([0, 0, -1.0]))
    out = np.cos(a_new)[:, None] * np.array([1.0, 0, 0]) + np.sin(a_new)[:, None] * perp
    return out * n


def targets(kp: dict, side: str, ik: ArmIk, pitch: float, k: float, c: np.ndarray,
            a0: float = 180.0, amax: float = 180.0):
    A = pitch_matrix(pitch)
    sh, el, wr = kp[f"{side}_shoulder"], kp[f"{side}_elbow"], kp[f"{side}_wrist"]
    wv = compress_cone((wr - sh) @ A.T, a0, amax)
    wt = ik.shoulder + c + k * wv
    u = compress_cone((el - sh) @ A.T, a0, amax)
    et = ik.shoulder + L_UPPER * u / np.linalg.norm(u, axis=1, keepdims=True)
    f = (wr - el) @ A.T
    hd = f / np.linalg.norm(f, axis=1, keepdims=True)
    return wt, et, hd


_NEUTRAL: dict = {}


def neutral_posture(arm):
    """Comfortable reference: wrist 0.3 m forward and 0.3 m below J2, clear of panel/table."""
    if arm not in _NEUTRAL:
        ik = ArmIk(arm)
        s1 = np.pi / 2 if arm == "left" else -np.pi / 2
        q0 = np.array([s1, 0.8, 0.0, -2.0, 0.0, 2.2, 0.0])
        wt = ik.shoulder + np.array([0.30, 0.0, -0.30])
        et = ik.shoulder + L_UPPER * np.array([0.3, 0.0, -0.95])
        q = ik.solve(q0, wt, et, np.array([1.0, 0, 0]), q0, iters=300)
        _NEUTRAL[arm] = q
    return _NEUTRAL[arm]


def _seeds(arm):
    s1 = np.pi / 2 if arm == "left" else -np.pi / 2
    return [np.array([s1 + d1, d2, d3, -1.8, 0, 2.0, 0.0])
            for d1 in (-1.2, 0, 1.2) for d2 in (-0.9, 0.0, 0.9) for d3 in (-0.8, 0.8)]


def seed_solve(ik: ArmIk, wt, et, hd, q_prev=None, q_post=None, cont_w=0.05):
    """Best IK over several seeds (plus q_prev); cost = wrist error + cont_w*|q - q_prev|."""
    seeds = _seeds(ik.arm) + ([q_prev] if q_prev is not None else [])
    best, bq = 1e9, None
    for q0 in seeds:
        q = ik.solve(q0, wt, et, hd, q0 if q_post is None else q_post, iters=150)
        cost = np.linalg.norm(ik.fk(q)[0] - wt)
        if q_prev is not None:
            cost += cont_w * np.linalg.norm(q - q_prev)
        if cost < best:
            best, bq = cost, q
    return bq


def run_clip(ik: ArmIk, kp: dict, side: str, pitch, k, c, stride=1, clearance=True,
             a0=180.0, amax=180.0, restart_tol=None):
    wt, et, hd = targets(kp, side, ik, pitch, k, np.asarray(c, float), a0, amax)
    idx = np.arange(0, len(wt), stride)
    q_post = neutral_posture(ik.arm)
    q = seed_solve(ik, wt[0], et[0], hd[0], q_post=q_post)
    Q, err, clr = [], [], []
    for i in idx:
        q_new = ik.solve(q, wt[i], et[i], hd[i], q_post)
        if restart_tol is not None and np.linalg.norm(ik.fk(q_new)[0] - wt[i]) > restart_tol:
            q_new = seed_solve(ik, wt[i], et[i], hd[i], q_prev=q, q_post=q_post)
        q = q_new
        Q.append(q)
        err.append(np.linalg.norm(ik.fk(q)[0] - wt[i]))
        clr.append(ik.clearance()[0] if clearance else np.nan)
    return np.array(Q), np.array(err), np.array(clr), idx


def _eval_combo(args):
    pitch, k, a0, amax, clips, stride = args
    out = {}
    iks = {s: ArmIk(s, weights=IK_WEIGHTS) for s in ("left", "right")}
    for clip in clips:
        kp = arm_keypoints(BVH_DIR / f"{clip}.bvh")
        for side in ("left", "right"):
            Q, err, clr, _ = run_clip(iks[side], kp, side, pitch, k, (0, 0, 0), stride=stride, a0=a0, amax=amax)
            sp = np.abs(np.diff(Q, axis=0)).max(axis=1) * kp["fps"] / stride
            out[f"{clip}/{side}"] = dict(track=float(np.mean(err < WRIST_TOL)),
                                         speed_p50=float(np.percentile(sp, 50)),
                                         speed_p90=float(np.percentile(sp, 90)),
                                         clear=float(np.mean(clr >= CLEARANCE_MIN)),
                                         ok=float(np.mean((err < WRIST_TOL) & (clr >= CLEARANCE_MIN))))
    return (pitch, k, a0, amax), out


def sweep(args):
    grid = list(itertools.product(args.pitch, args.k, args.a0, args.amax))
    jobs = [(p, k, a0, am, CANDIDATES, args.stride) for p, k, a0, am in grid]
    results = {}
    with ProcessPoolExecutor(max_workers=min(len(jobs), args.workers)) as ex:
        for key, out in ex.map(_eval_combo, jobs):
            results[key] = out
            ok = np.mean([v["ok"] for v in out.values()])
            tr = np.mean([v["track"] for v in out.values()])
            cl = np.mean([v["clear"] for v in out.values()])
            s50 = np.median([v["speed_p50"] for v in out.values()])
            s90 = np.median([v["speed_p90"] for v in out.values()])
            print(f"pitch={key[0]:>4} k={key[1]:.2f} a0={key[2]:>4} amax={key[3]:>4}  ok={ok:.3f} track={tr:.3f} clear={cl:.3f} speed50={s50:.2f} speed90={s90:.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--process", action="store_true", help="full-rate retarget of CANDIDATES with CHOSEN")
    ap.add_argument("--pitch", type=float, nargs="+", default=[0, 25, 45])
    ap.add_argument("--k", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    ap.add_argument("--a0", type=float, nargs="+", default=[30, 50])
    ap.add_argument("--amax", type=float, nargs="+", default=[70, 80])
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "panda_retarget" / "sweep.json")
    a = ap.parse_args()
    if a.sweep:
        sweep(a)
    if a.process:
        process(a)



# --------------------------------------------------------------------------
# Full-rate processing of the chosen mapping
# --------------------------------------------------------------------------
CHOSEN = dict(pitch=0.0, k=1.0, a0=30.0, amax=50.0)   # from runs/panda_retarget/sweep*.log
SPEED_REF = 0.3        # target peak joint speed of the reference (0.6 x 0.5 rad/s cap)
MIN_SEG_S = 5.0


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)


def segments(mask: np.ndarray, min_len: int):
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def _process_clip(clip: str):
    kp = arm_keypoints(BVH_DIR / f"{clip}.bvh")
    fps = kp["fps"]
    iks = {s: ArmIk(s, weights=IK_WEIGHTS) for s in ("left", "right")}
    Q, err = {}, {}
    for side in ("left", "right"):
        q, e, _, _ = run_clip(iks[side], kp, side, CHOSEN["pitch"], CHOSEN["k"], (0, 0, 0),
                              stride=1, clearance=False, a0=CHOSEN["a0"], amax=CHOSEN["amax"])
        Q[side] = gaussian_smooth(q, 2.0)
        err[side] = e
    L, R = iks["left"], iks["right"]
    m, d = L.m, L.d
    T = len(Q["left"])
    clr = {"left": np.zeros(T), "right": np.zeros(T)}
    who = {"left": [], "right": []}
    inter = np.zeros(T)
    ft = np.zeros(6)
    for t in range(T):
        d.qpos[L.qadr] = Q["left"][t]
        d.qpos[R.qadr] = Q["right"][t]
        mujoco.mj_kinematics(m, d)
        for side, ik in (("left", L), ("right", R)):
            ik.d.qpos[:] = d.qpos
            mujoco.mj_kinematics(ik.m, ik.d)
            c, w = ik.env_clearance()
            clr[side][t] = c
            who[side].append(w)
        inter[t] = min(mujoco.mj_geomDistance(m, d, a, b, 0.3, ft) for a in L.arm_geoms for b in R.arm_geoms)
    valid = (clr["left"] >= CLEARANCE_MIN) & (clr["right"] >= CLEARANCE_MIN) & (inter >= CLEARANCE_MIN)
    Qb = np.concatenate([Q["left"], Q["right"]], axis=1)
    qd = np.gradient(Qb, axis=0) * fps
    segs = []
    for a, b in segments(valid, int(MIN_SEG_S * fps)):
        peak = float(np.abs(qd[a:b]).max())
        segs.append(dict(start=a, end=b, seconds=(b - a) / fps, peak_speed=peak,
                         p99_speed=float(np.percentile(np.abs(qd[a:b]).max(axis=1), 99)),
                         slowdown=max(1.0, peak / SPEED_REF)))
    out_dir = REPO / "runs" / "panda_retarget" / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{clip}.npz", q=Qb, fps=fps, valid=valid, clr_left=clr["left"],
                        clr_right=clr["right"], inter=inter, err_left=err["left"], err_right=err["right"])
    fails = {}
    for side in ("left", "right"):
        bad = [w for w, c in zip(who[side], clr[side]) if c < CLEARANCE_MIN]
        for w in set(bad):
            fails[f"{side}:{w}"] = bad.count(w) / T
    fails["inter_arm"] = float(np.mean(inter < CLEARANCE_MIN))
    return clip, dict(frames=T, seconds=T / fps, valid=float(valid.mean()),
                      track_left=float(np.mean(err["left"] < WRIST_TOL)),
                      track_right=float(np.mean(err["right"] < WRIST_TOL)),
                      fails=fails, segments=segs)


def process(args):
    summary = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for clip, s in ex.map(_process_clip, CANDIDATES):
            summary[clip] = s
            seg_s = sum(x["seconds"] for x in s["segments"])
            print(f"{clip:28s} {s['seconds']:6.1f}s valid={s['valid']:.2f} segs={len(s['segments'])} "
                  f"seg_total={seg_s:6.1f}s fails={ {k: round(v, 2) for k, v in s['fails'].items() if v > 0.01} }", flush=True)
    (REPO / "runs" / "panda_retarget" / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
