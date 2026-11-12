"""Export the source LAFAN1 human motion for each deployed robot reference (for side-by-side viewing).

For every franka_mimic/refs/<clip>_segNN.npz this recovers, frame by frame, the source BVH
time that the robot reference shows, through the same two time maps the references went through:
  1. adaptive time warp (tools/build_panda_motions.py warp_segment), 30 fps source -> 50 Hz
  2. raised-cosine ease in/out + end hold (tools/export_deploy.py ease)
and writes the full skeleton in the actor's torso frame (+x forward, +y actor's left, +z up,
origin between the shoulders; the same frame the arm retargeting uses) to
franka_mimic/refs/human/<name>.npz. Both maps are re-run here and checked against the stored
robot references (max |q| difference printed).

LAFAN1 is CC BY-NC-ND: the output stays local (franka_mimic/refs/ is gitignored).

    uv run python tools/export_human_refs.py [--out ~/harry/franka_mimic]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import panda_sets
from build_panda_motions import CONTROL_DT, V_REF, _maxfilt, _smooth
from bvh_arms import _Y_TO_Z_UP, global_positions, read_bvh
from retarget_bvh_arms_to_panda import BVH_DIR

DT = 0.02
EASE_S = 1.5  # export_deploy.py default --ease


def warp_times(q: np.ndarray, fps: float, v_ref: float = V_REF, iters: int = 8):
    """build_panda_motions.warp_segment, returning the source time of every 50 Hz sample too."""
    T = len(q)
    t_src = np.arange(T) / fps
    speed = np.abs(np.gradient(q, axis=0)).max(axis=1) * fps
    f = np.maximum(1.0, speed / v_ref)
    win = int(0.25 * fps)
    for _ in range(iters):
        g = np.maximum(1.0, _smooth(_maxfilt(f, win), 0.25 * fps))
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) / fps)])
        n = int(np.floor(tau[-1] / CONTROL_DT)) + 1
        t_new = np.interp(np.arange(n) * CONTROL_DT, tau, t_src)
        q50 = np.stack([np.interp(t_new, t_src, q[:, j]) for j in range(q.shape[1])], axis=1)
        qd50 = np.gradient(q50, CONTROL_DT, axis=0)
        if np.abs(qd50).max() <= v_ref * 1.01:
            break
        over_src = np.interp(t_src, t_new, np.abs(qd50).max(axis=1) / v_ref)
        f = f * np.maximum(1.0, over_src)
    return q50, t_new


def ease_times(n: int, seconds: float = EASE_S) -> np.ndarray:
    """export_deploy.ease, returning the (fractional) source sample index of every output tick."""
    t_end = (n - 1) * DT
    taus, ts = [0.0], [0.0]
    while ts[-1] < t_end:
        tau = taus[-1]
        s_in = 0.5 - 0.5 * np.cos(np.pi * min(tau / seconds, 1.0))
        s_out = 0.5 - 0.5 * np.cos(np.pi * min((t_end - ts[-1]) / (0.5 * seconds), 1.0))
        s = max(min(s_in, s_out), 0.02)
        taus.append(tau + DT)
        ts.append(min(ts[-1] + s * DT, t_end))
    ts = np.array(ts)
    ts = np.concatenate([ts, np.full(int(seconds / DT) // 2, ts[-1])])
    return ts / DT


def torso_frame_skeleton(clip: str):
    b = read_bvh(BVH_DIR / f"{clip}.bvh")
    P = global_positions(b) @ _Y_TO_Z_UP.T
    g = {n: P[:, b.index(n)] for n in ("Spine", "Neck", "LeftArm", "RightArm")}
    left = g["LeftArm"] - g["RightArm"]
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    up = g["Neck"] - g["Spine"]
    up -= np.sum(up * left, axis=1, keepdims=True) * left
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    R = np.stack([np.cross(left, up), left, up], axis=2)
    origin = 0.5 * (g["LeftArm"] + g["RightArm"])
    S = np.einsum("tji,tkj->tki", R, P - origin[:, None])
    return S, b.names, b.parents, 1.0 / b.frame_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("~/harry/franka_mimic"))
    panda_sets.add_args(ap)
    a = ap.parse_args()
    rs = panda_sets.get(a.set)
    RET = rs.ret
    refs = a.out.expanduser() / "refs" / rs.name
    (refs / "human").mkdir(parents=True, exist_ok=True)
    summary = json.loads((RET / "summary.json").read_text())
    cache = {}
    for ref in sorted(refs.glob("*_seg*.npz")):
        clip, seg = ref.stem.rsplit("_seg", 1)
        if not seg.isdigit():      # windows cut by export_ref_window.py write their own skeleton
            continue
        s = summary[clip]["segments"][int(seg)]
        cz = np.load(RET / "clips" / f"{clip}.npz")
        fps = float(cz["fps"])
        q50, t_new = warp_times(cz["q"][s["start"]:s["end"]], fps)
        k = ease_times(len(q50))
        robot = np.load(ref)["q"]
        q_chk = np.stack([np.interp(k, np.arange(len(q50)), q50[:, j]) for j in range(14)], 1)
        n = min(len(robot), len(q_chk))
        err = np.abs(q_chk[:n] - robot[:n]).max()
        src_frame = s["start"] + np.interp(k, np.arange(len(t_new)), t_new) * fps
        if clip not in cache:
            cache[clip] = torso_frame_skeleton(clip)
        S, names, parents, _ = cache[clip]
        i0 = np.clip(np.floor(src_frame).astype(int), 0, len(S) - 2)
        w = (src_frame - i0)[:, None, None]
        pos = (1 - w) * S[i0] + w * S[i0 + 1]
        np.savez(refs / "human" / f"{ref.stem}.npz", pos=pos[: len(robot)].astype(np.float32),
                 parents=parents, names=np.array(names), src_frame=src_frame[: len(robot)], dt=DT)
        print(f"{ref.stem}: {len(robot)} ticks ({len(k)} mapped), source frames {src_frame[0]:.0f}-{src_frame[-1]:.0f}, "
              f"robot-ref check max|dq| = {err:.2e} rad")


if __name__ == "__main__":
    main()
