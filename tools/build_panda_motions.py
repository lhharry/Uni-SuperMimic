"""Retime retargeted dual-Panda segments and export Uni-SuperMimic compact motion packs.

Input: runs/panda_retarget/{summary.json, clips/<clip>.npz} from
retarget_bvh_arms_to_panda.py --process (30 fps, both arms, validity mask).

Retiming (adaptive time warp): each source frame is stretched by
f_t = max(1, |qd|_inf(t) / v_ref), max-filtered and smoothed so the tempo
changes gradually, then the segment is resampled at 50 Hz in warped time.
Iterated until the peak joint speed is <= v_ref (0.3 rad/s = 0.6 x the
0.5 rad/s deployment cap).

Output (gitignored; LAFAN1 is CC BY-NC-ND, do not redistribute derivatives):
  src/unisupermimic/assets/motions/panda/lafan1_panda_dual.pt  (J=14)
  src/unisupermimic/assets/motions/panda/lafan1_panda_left.pt  (J=7, left arm)
Compact layout per row: [root_p 3, root_quat_wxyz 4, q J, root_v 3, root_w 3, qd J]
with a constant root (fixed base).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RET = REPO / "runs" / "panda_retarget"
OUT = REPO / "src" / "unisupermimic" / "assets" / "motions" / "panda"
CONTROL_DT = 0.02
V_REF = 0.3


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def _maxfilt(x: np.ndarray, r: int) -> np.ndarray:
    xp = np.pad(x, r, mode="edge")
    return np.max(np.lib.stride_tricks.sliding_window_view(xp, 2 * r + 1), axis=1)


def warp_segment(q: np.ndarray, fps: float, v_ref: float = V_REF, iters: int = 8):
    """Return (q50, qd50, factor) resampled at 50 Hz with peak |qd| <= v_ref."""
    T = len(q)
    t_src = np.arange(T) / fps
    speed = np.abs(np.gradient(q, axis=0)).max(axis=1) * fps
    f = np.maximum(1.0, speed / v_ref)
    win = int(0.25 * fps)
    for _ in range(iters):
        g = np.maximum(1.0, _smooth(_maxfilt(f, win), 0.25 * fps))
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) / fps)])
        n = int(np.floor(tau[-1] / CONTROL_DT)) + 1
        tau_new = np.arange(n) * CONTROL_DT
        t_new = np.interp(tau_new, tau, t_src)
        q50 = np.stack([np.interp(t_new, t_src, q[:, j]) for j in range(q.shape[1])], axis=1)
        qd50 = np.gradient(q50, CONTROL_DT, axis=0)
        peak = np.abs(qd50).max()
        if peak <= v_ref * 1.01:
            break
        # stretch more where the result is still too fast
        over = np.abs(qd50).max(axis=1) / v_ref
        over_src = np.interp(t_src, t_new, over)
        f = f * np.maximum(1.0, over_src)
    return q50, qd50, tau[-1] / t_src[-1]


def pack(segs: list[tuple[str, np.ndarray, np.ndarray]], J: int) -> dict:
    rows, names, starts, lengths, s = [], [], [], [], 0
    for name, q, qd in segs:
        n = len(q)
        root = np.zeros((n, 7))
        root[:, 3] = 1.0                                  # quat wxyz identity
        rows.append(np.concatenate([root, q[:, :J], np.zeros((n, 6)), qd[:, :J]], axis=1))
        names.append(name)
        starts.append(s)
        lengths.append(n)
        s += n
    ref = torch.tensor(np.concatenate(rows), dtype=torch.float32)
    assert ref.shape[1] == 13 + 2 * J
    return {"ref": ref, "names": names, "starts": starts, "lengths": lengths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-ref", type=float, default=V_REF)
    a = ap.parse_args()
    summary = json.loads((RET / "summary.json").read_text())
    segs, report = [], []
    for clip, s in summary.items():
        z = np.load(RET / "clips" / f"{clip}.npz")
        q, fps = z["q"], float(z["fps"])
        for i, seg in enumerate(s["segments"]):
            q50, qd50, factor = warp_segment(q[seg["start"]:seg["end"]], fps, a.v_ref)
            name = f"{clip}_seg{i:02d}"
            segs.append((name, q50, qd50))
            report.append(dict(name=name, src_s=seg["seconds"], out_s=len(q50) * CONTROL_DT,
                               factor=factor, peak=float(np.abs(qd50).max())))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(pack(segs, 14), OUT / "lafan1_panda_dual.pt")
    left = [(n, q[:, :7], qd[:, :7]) for n, q, qd in segs]
    torch.save(pack(left, 7), OUT / "lafan1_panda_left.pt")
    (RET / "motions_report.json").write_text(json.dumps(report, indent=1))
    src = sum(r["src_s"] for r in report)
    out = sum(r["out_s"] for r in report)
    fac = np.array([r["factor"] for r in report])
    print(f"{len(report)} segments: source {src / 60:.1f} min -> retimed {out / 60:.1f} min "
          f"(x{out / src:.2f}); per-segment factor p10/50/90 = {np.percentile(fac, [10, 50, 90]).round(2)}; "
          f"max peak {max(r['peak'] for r in report):.3f} rad/s")


if __name__ == "__main__":
    main()
