"""Cut a source-time window out of one retimed segment and export it as an extra deploy reference.

Same processing as tools/export_deploy.py (raised-cosine ease in/out + end hold, goal features,
clearance check) plus the matching human skeleton (tools/export_human_refs.py), so the robot
side (replay.py, view_ref.py) treats it like any other reference.

    uv run python tools/export_ref_window.py dance2_subject1_seg00 --src 9.5 16.5 \
        --name dance2_subject1_seg00w095-165 [--out ~/harry/franka_mimic]

The name must keep the '<clip>_seg' prefix (view_ref.py finds the G1 clip from it).
LAFAN1 is CC BY-NC-ND: outputs stay local (franka_mimic/refs/ is gitignored).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import panda_sets
from export_deploy import DT, clearances, ease, goal_features
from export_human_refs import ease_times, torso_frame_skeleton, warp_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segment", help="e.g. dance2_subject1_seg00")
    ap.add_argument("--src", nargs=2, type=float, required=True, metavar=("FROM_S", "TO_S"),
                    help="window in the source clip's time (seconds, 30 fps BVH)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ease", type=float, default=1.5)
    ap.add_argument("--out", type=Path, default=Path("~/harry/franka_mimic"))
    panda_sets.add_args(ap)      # adds a reference to that set's deploy dir; never touches its packs
    a = ap.parse_args()
    rs = panda_sets.get(a.set)
    RET, PACK = rs.ret, rs.pack("dual")
    out = a.out.expanduser()
    refs = out / "refs" / rs.name
    clip, seg = a.segment.rsplit("_seg", 1)
    if not a.name.startswith(f"{clip}_seg"):
        raise SystemExit(f"--name must start with {clip}_seg")

    s = json.loads((RET / "summary.json").read_text())[clip]["segments"][int(seg)]
    cz = np.load(RET / "clips" / f"{clip}.npz")
    fps = float(cz["fps"])
    q50, t_new = warp_times(cz["q"][s["start"]:s["end"]], fps)
    import torch
    pack = torch.load(PACK, weights_only=True)
    i = pack["names"].index(a.segment)
    q_pack = pack["ref"][pack["starts"][i]:pack["starts"][i] + pack["lengths"][i], 7:21].numpy().astype(np.float64)
    assert len(q_pack) == len(q50) and np.abs(q_pack - q50).max() < 1e-4, "warp does not reproduce the pack"

    src_s = s["start"] / fps + t_new                       # source time of every 50 Hz sample
    lo, hi = np.searchsorted(src_s, a.src[0]), np.searchsorted(src_s, a.src[1])
    if hi - lo < 50:
        raise SystemExit(f"window outside segment ({src_s[0]:.1f}-{src_s[-1]:.1f} s source)")
    q = q_pack[lo:hi]
    qe, qde = ease(q, a.ease)
    env, inter = clearances(qe)
    np.savez(refs / f"{a.name}.npz", q=qe.astype(np.float32), qd=qde.astype(np.float32), dt=DT,
             goal_left=goal_features(qe, qde, "left"), goal_dual=goal_features(qe, qde, "dual"))

    # matching human skeleton: eased tick -> window sample -> segment sample -> source frame
    k = ease_times(len(q), a.ease)[: len(qe)] + lo
    src_frame = s["start"] + np.interp(k, np.arange(len(t_new)), t_new) * fps
    S, names, parents, _ = torso_frame_skeleton(clip)
    i0 = np.clip(np.floor(src_frame).astype(int), 0, len(S) - 2)
    w = (src_frame - i0)[:, None, None]
    (refs / "human").mkdir(parents=True, exist_ok=True)
    np.savez(refs / "human" / f"{a.name}.npz", pos=((1 - w) * S[i0] + w * S[i0 + 1]).astype(np.float32),
             parents=parents, names=np.array(names), src_frame=src_frame, dt=DT)

    row = dict(name=a.name, seconds=round(len(qe) * DT, 1), min_env_clearance=round(env, 3),
               min_inter_arm=round(inter, 3), peak_speed=round(float(np.abs(qde).max()), 3),
               left_j1_range=round(float(np.ptp(qe[:, 0])), 2),
               source=f"{clip} {a.src[0]:g}-{a.src[1]:g} s (window of {a.segment})")
    idx_path = refs / "index.json"
    index = [r for r in json.loads(idx_path.read_text()) if r["name"] != a.name]
    idx_path.write_text(json.dumps(index + [row], indent=1))
    print(row)
    print("per-arm joint ranges (rad): left", np.round(np.ptp(qe[:, :7], 0), 2).tolist(),
          " right", np.round(np.ptp(qe[:, 7:], 0), 2).tolist())


if __name__ == "__main__":
    main()
