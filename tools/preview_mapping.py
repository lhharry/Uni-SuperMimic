"""Stage 5.6: preview references of a candidate mapping for the viewer (franka_mimic view_ref.py).

Retargets the clips of the given source-time windows with the mapping (the real pipeline,
full rate), cuts the windows, retimes them to 0.3 rad/s and eases them exactly like the deploy
references, and writes refs/<set>/<clip>_segW<from>-<to>.npz plus the matching human skeleton
(refs/<set>/human/). Same window names for every candidate, so the viewer can show them one
after another (or side by side with --sync-source).

    uv run python tools/preview_mapping.py --set cand_A --sweep-index 37
    uv run python tools/preview_mapping.py --set v1b --v1
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retarget_bvh_arms_to_panda as rt  # noqa: E402
from export_deploy import DT, ease  # noqa: E402
from export_human_refs import ease_times, torso_frame_skeleton, warp_times  # noqa: E402

WINDOWS = [("dance2_subject1", 9.5, 16.5),           # the wave (also cut from v1)
           ("dance2_subject4", 0.0, 7.6),            # v1 dance2_subject4_seg00, run on the robots
           ("dance2_subject3", 75.0, 83.0),          # mostly sideways / raised arms
           ("fightAndSports1_subject4", 0.0, 6.1)]   # v1 fightAndSports1_subject4_seg00


def _clip(job):
    clip, m = job
    r = rt.retarget_clip(clip, m, stride=1)
    return clip, r["Q"], r["valid"], r["fps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep-index", type=int)
    g.add_argument("--mapping", type=Path)
    g.add_argument("--v1", action="store_true", help="the v1 cone (= v1b on the new pipeline)")
    ap.add_argument("--sweep", type=Path, default=rt.REPO / "runs" / "retarget_sweep_stage5.json")
    ap.add_argument("--out", type=Path, default=Path("~/harry/franka_mimic"))
    a = ap.parse_args()
    if a.v1:
        m = rt.V1
    elif a.mapping:
        m = rt.Mapping(**json.loads(a.mapping.read_text()))
    else:
        m = rt.Mapping(**json.loads(a.sweep.read_text())["results"][a.sweep_index]["mapping"])
    refs = a.out.expanduser() / "refs" / a.set
    (refs / "human").mkdir(parents=True, exist_ok=True)
    clips = sorted({w[0] for w in WINDOWS})
    with ProcessPoolExecutor(len(clips)) as ex:
        res = {c: (Q, v, fps) for c, Q, v, fps in ex.map(_clip, [(c, m) for c in clips])}
    index = []
    for clip, t0, t1 in WINDOWS:
        Q, valid, fps = res[clip]
        lo, hi = int(round(t0 * fps)), int(round(t1 * fps))
        q50, t_new = warp_times(Q[lo:hi], fps)
        qe, qde = ease(q50, 1.5)
        name = f"{clip}_segW{t0:05.1f}-{t1:05.1f}".replace(".", "")
        np.savez(refs / f"{name}.npz", q=qe.astype(np.float32), qd=qde.astype(np.float32), dt=DT)
        k = ease_times(len(q50), 1.5)[: len(qe)]
        src = lo + np.interp(k, np.arange(len(t_new)), t_new) * fps
        S, names, parents, _ = torso_frame_skeleton(clip)
        i0 = np.clip(np.floor(src).astype(int), 0, len(S) - 2)
        w = (src - i0)[:, None, None]
        np.savez(refs / "human" / f"{name}.npz", pos=((1 - w) * S[i0] + w * S[i0 + 1]).astype(np.float32),
                 parents=parents, names=np.array(names), src_frame=src, dt=DT)
        row = dict(name=name, seconds=round(len(qe) * DT, 1), source=f"{clip} {t0}-{t1} s",
                   valid_frac=round(float(valid[lo:hi].mean()), 2), slowdown=round(len(qe) * DT / (t1 - t0), 1))
        index.append(row)
        print(row)
    (refs / "index.json").write_text(json.dumps(index, indent=1))
    (refs / "mapping.json").write_text(json.dumps(asdict(m), indent=1))
    print(f"wrote {refs}")


if __name__ == "__main__":
    main()
