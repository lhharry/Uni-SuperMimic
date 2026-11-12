"""Human-likeness of retargeted Panda arm motion, measured on the actual FK (stage 5.3).

The retarget sweep's 'track' scores the distance to the *compressed* wrist target, which
rewards compression. These metrics compare the Panda with the *uncompressed* human instead,
both in the robot/torso frame (+x out of the panel = forward, +y = left, +z = up):

  D_w        angle between Panda J2-centre -> link6 and human shoulder -> wrist  (main metric)
  D_e        angle between Panda J2-centre -> link4 and human shoulder -> elbow (reported only)
  elev       angle of each of those wrist directions from +x; corr and spread ratio
             (Panda p90-p10 / human p90-p10) say whether the motion got "smaller"
  sectors    frames split by the human wrist direction: fwd (< 30 deg from +x), else by the
             dominant component: down / up / out (away from the other arm) / in

    uv run python tools/human_likeness.py --set v1                 # metrics of an existing set
    uv run python tools/human_likeness.py --upper-bound            # no cone, no IK barrier
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panda_sets  # noqa: E402
from bvh_arms import arm_keypoints  # noqa: E402
from panda_arm_ik import ArmIk, IkWeights, inter_arm_distance  # noqa: E402

BVH_DIR = panda_sets.REPO / "data" / "lafan1_bvh"
SECTORS = ("fwd", "down", "out", "up", "in")
CLEAR = 0.15


def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


def sector(u: np.ndarray, side: str) -> np.ndarray:
    """Sector of each human wrist direction (rows of u, torso frame)."""
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    out_sign = 1.0 if side == "left" else -1.0
    fwd = np.degrees(np.arccos(np.clip(u[:, 0], -1, 1))) < 30
    comp = np.stack([-u[:, 2], out_sign * u[:, 1], u[:, 2], -out_sign * u[:, 1]], 1)   # down out up in
    lab = np.array(SECTORS[1:])[comp.argmax(1)]
    return np.where(fwd, "fwd", lab)


def arm_fk(ik: ArmIk, Q: np.ndarray):
    W, E = np.zeros((len(Q), 3)), np.zeros((len(Q), 3))
    for i, q in enumerate(Q):
        W[i], E[i], _ = ik.fk(q)
    return W - ik.shoulder, E - ik.shoulder


def frame_metrics(kp: dict, side: str, ik: ArmIk, Q: np.ndarray, idx: np.ndarray) -> dict:
    pw, pe = arm_fk(ik, Q)
    hw = kp[f"{side}_wrist"][idx] - kp[f"{side}_shoulder"][idx]
    he = kp[f"{side}_elbow"][idx] - kp[f"{side}_shoulder"][idx]
    ex = np.array([[1.0, 0, 0]])
    return dict(D_w=_ang(pw, hw), D_e=_ang(pe, he), elev_p=_ang(pw, ex.repeat(len(pw), 0)),
                elev_h=_ang(hw, ex.repeat(len(hw), 0)), sector=sector(hw, side),
                reach=np.linalg.norm(pw, axis=1) / np.linalg.norm(hw, axis=1))


def summarise(rows: list[dict], mask_key: str | None = None) -> dict:
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0] if k != "clip"}
    m = cat[mask_key].astype(bool) if mask_key else np.ones(len(cat["D_w"]), bool)
    out = dict(frames=int(m.sum()), frac=round(float(m.mean()), 3))
    d, sec = cat["D_w"][m], cat["sector"][m]
    out["D_w_p50"], out["D_w_p90"] = (round(float(np.percentile(d, p)), 1) for p in (50, 90))
    out["D_e_p50"] = round(float(np.percentile(cat["D_e"][m], 50)), 1)
    ep, eh = cat["elev_p"][m], cat["elev_h"][m]
    out["elev_p50_panda_human"] = [round(float(np.percentile(ep, 50)), 1), round(float(np.percentile(eh, 50)), 1)]
    out["elev_corr"] = round(float(np.corrcoef(ep, eh)[0, 1]), 2)
    out["elev_spread_ratio"] = round(float(np.ptp(np.percentile(ep, [10, 90])) / np.ptp(np.percentile(eh, [10, 90]))), 2)
    out["reach_p50"] = round(float(np.percentile(cat["reach"][m], 50)), 2)
    out["sectors"] = {s: dict(frac=round(float((sec == s).mean()), 3),
                              D_w_p50=round(float(np.percentile(d[sec == s], 50)), 1) if (sec == s).any() else None,
                              D_w_p90=round(float(np.percentile(d[sec == s], 90)), 1) if (sec == s).any() else None)
                      for s in SECTORS}
    return out


def _set_clip(job):
    clip, ret, stride = job
    z = np.load(ret / "clips" / f"{clip}.npz")
    kp = arm_keypoints(BVH_DIR / f"{clip}.bvh")
    idx = np.arange(0, len(z["q"]), stride)
    rows = []
    for k, side in enumerate(("left", "right")):
        ik = ArmIk(side)
        r = frame_metrics(kp, side, ik, z["q"][idx, 7 * k:7 * k + 7], idx)
        r["valid"] = z["valid"][idx]
        r["track"] = z[f"err_{side}"][idx] < 0.04
        r["valid_track"] = r["valid"] & r["track"]
        rows.append(r)
    return rows


def _upper_clip(job):
    """No cone, no IK collision barrier; feasibility (clearance, inter-arm) measured afterwards."""
    clip, stride = job
    import retarget_bvh_arms_to_panda as rt
    kp = arm_keypoints(BVH_DIR / f"{clip}.bvh")
    w = IkWeights(elbow=rt.IK_WEIGHTS.elbow, posture=rt.IK_WEIGHTS.posture, step=rt.IK_WEIGHTS.step, collision=0.0)
    iks = {s: ArmIk(s, weights=w) for s in ("left", "right")}
    Q, E = {}, {}
    for side in ("left", "right"):
        Q[side], E[side], _, _ = rt.run_clip(iks[side], kp, side, 0.0, 1.0, (0, 0, 0), stride=stride,
                                             clearance=False, a0=180.0, amax=180.0)
        Q[side] = np.asarray(Q[side])
    idx = np.arange(0, len(kp["left_wrist"]), stride)
    L, R = iks["left"], iks["right"]
    clr = {"left": np.zeros(len(idx)), "right": np.zeros(len(idx))}
    inter = np.zeros(len(idx))
    for t in range(len(idx)):
        L.set_q(Q["left"][t])
        R.set_q(Q["right"][t])
        clr["left"][t], clr["right"][t] = L.env_clearance()[0], R.env_clearance()[0]
        inter[t] = inter_arm_distance(L, R)
    rows = []
    for side in ("left", "right"):
        r = frame_metrics(kp, side, iks[side], Q[side], idx)
        r["track"] = np.asarray(E[side]) < 0.04
        r["env_ok"] = clr[side] >= CLEAR
        r["feasible"] = r["env_ok"] & (inter >= CLEAR)
        r["feasible_track"] = r["feasible"] & r["track"]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None)
    ap.add_argument("--upper-bound", action="store_true")
    ap.add_argument("--clips", nargs="+", default=None, help="default: the retarget CANDIDATES")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import retarget_bvh_arms_to_panda as rt
    clips = a.clips or list(rt.CANDIDATES)
    with ProcessPoolExecutor(a.workers) as ex:
        if a.upper_bound:
            rows = [r for rr in ex.map(_upper_clip, [(c, a.stride) for c in clips]) for r in rr]
            res = {"mode": "upper bound (no cone, no IK collision barrier)", "clips": clips,
                   "all": summarise(rows), "feasible": summarise(rows, "feasible"),
                   "feasible_track": summarise(rows, "feasible_track")}
            out = a.out or panda_sets.REPO / "runs" / "human_likeness_upper_bound.json"
        else:
            rs = panda_sets.get(a.set or "v1")
            rows = [r for rr in ex.map(_set_clip, [(c, rs.ret, a.stride) for c in clips]) for r in rr]
            res = {"mode": f"set {rs.name}", "clips": clips, "all": summarise(rows),
                   "valid": summarise(rows, "valid"), "valid_track": summarise(rows, "valid_track")}
            out = a.out or rs.ret / "human_likeness.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if isinstance(v, dict):
            sec = " ".join(f"{s}:{d['frac']:.2f}/{d['D_w_p50']}" for s, d in v["sectors"].items())
            print(f"{k:15s} frac={v['frac']:.2f} D_w p50/p90={v['D_w_p50']}/{v['D_w_p90']}  D_e p50={v['D_e_p50']}  "
                  f"elev p50 panda/human={v['elev_p50_panda_human']} corr={v['elev_corr']} spread={v['elev_spread_ratio']}  "
                  f"reach={v['reach_p50']}\n{'':15s} sectors frac/D_w_p50: {sec}")
    print(out)


if __name__ == "__main__":
    main()
