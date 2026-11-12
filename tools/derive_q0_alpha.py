"""Derive the policy action frame q0 / alpha of a reference set from its dual motion pack.

The rule (reconstructed 2026-11-12; reproduces the v1 constants in robot.py to 5e-4 rad):
  q0    = per joint, midpoint of the 0.5th and 99.5th percentile over all frames
  alpha = per joint, max |q - q0| over all frames / 4    (so |a| <= 4 covers every reference)

    uv run python tools/derive_q0_alpha.py --set v1 --check      # compare with robot.py
    uv run python tools/derive_q0_alpha.py --set v2              # writes lafan1_panda_v2.q0alpha.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import panda_sets
import torch

from unisupermimic.tasks.motion_tracking.panda.supermimic import robot as R


def derive(pack_path) -> tuple[dict, dict]:
    p = torch.load(pack_path, weights_only=True)
    q = p["ref"][:, 7:21].numpy().astype(np.float64)
    q0 = 0.5 * (np.percentile(q, 0.5, axis=0) + np.percentile(q, 99.5, axis=0))
    alpha = np.abs(q - q0).max(axis=0) / 4.0
    split = lambda v: {"left": np.round(v[:7], 3).tolist(), "right": np.round(v[7:], 3).tolist()}  # noqa: E731
    return split(q0), split(alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--check", action="store_true", help="only compare with robot.py (v1)")
    a = ap.parse_args()
    s = panda_sets.get(a.set)
    q0, alpha = derive(s.pack("dual"))
    if a.check or a.set == "v1":
        for name, mine, ref in (("q0", q0, R.Q0), ("alpha", alpha, R.ALPHA)):
            err = max(abs(x - y) for arm in ("left", "right") for x, y in zip(mine[arm], ref[arm]))
            print(f"{name}: max |derived - robot.py| = {err:.4f} rad")
        return
    s.q0alpha.write_text(json.dumps({"set": a.set, "rule": "q0=mid(p0.5,p99.5); alpha=max|q-q0|/4",
                                     "q0": q0, "alpha": alpha}, indent=1))
    print(f"wrote {s.q0alpha.relative_to(panda_sets.REPO)}\nq0 {q0}\nalpha {alpha}")


if __name__ == "__main__":
    main()
