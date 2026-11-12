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

Stage 5: the mapping is a Mapping (cone = v1, or anisotropic per direction), and the
sweep (tools/retarget_sweep.py) runs exactly this pipeline (retarget_clip) on a grid.

    uv run python tools/retarget_bvh_arms_to_panda.py --process --set v2 --mapping m.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panda_sets  # noqa: E402
from bvh_arms import arm_keypoints  # noqa: E402
from panda_arm_ik import ArmIk, IkWeights, inter_arm_distance  # noqa: E402

RET_DIR = panda_sets.get("v1").ret   # set in main() from --set

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
IK_WEIGHTS = IkWeights(elbow=0.05, posture=0.03, step=0.2)   # wrist-led; smooth through J2~0
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


@dataclass(frozen=True)
class Mapping:
    """Direction mapping + IK barrier settings of one retarget (stage 5).

    kind="cone":  compress_cone(a0, amax), the v1 mapping.
    kind="aniso": polar angle theta from +x is kept up to a0, beyond it saturates softly toward
                  amax(phi): theta' = a0 + (amax - a0) tanh((theta - a0) / (amax - a0)); amax >= 179
                  means no squeeze in that direction. amax(phi) blends the four sector values
                  (out = away from the other arm, up, in, down) with cos^2 weights, so adjacent
                  sectors always sum to one.
    wrist_tol_valid: frames whose wrist misses its (mapped) target by more than this are invalid
                  (None = v1 behaviour, clearance only).
    """
    kind: str = "aniso"
    a0: float = 30.0
    amax: float = 50.0
    out: float = 90.0
    up: float = 90.0
    inn: float = 50.0
    down: float = 50.0
    pitch: float = 0.0
    k: float = 1.0
    collision: float = 2.0
    safe_dist: float = 0.17
    soft_margin: float = 0.45
    posture: float = 0.03
    wrist_tol_valid: float | None = None   # stage 5: wrist error is reported (D_w), not cut

    def weights(self) -> IkWeights:
        return IkWeights(elbow=IK_WEIGHTS.elbow, posture=self.posture, step=IK_WEIGHTS.step,
                         collision=self.collision, safe_dist=self.safe_dist, soft_margin=self.soft_margin)


V1 = Mapping(kind="cone", a0=30.0, amax=50.0, wrist_tol_valid=None)   # = the old CHOSEN


def map_dirs(v: np.ndarray, side: str, m: Mapping) -> np.ndarray:
    if m.kind == "cone":
        return compress_cone(v, m.a0, m.amax)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    u = v / n
    th = np.arccos(np.clip(u[:, 0], -1, 1))
    s = 1.0 if side == "left" else -1.0              # +y is "out" for the left arm
    phi = np.arctan2(u[:, 2], s * u[:, 1])           # out 0, up +90, in 180, down -90 deg
    w = lambda c: np.maximum(np.cos(phi - np.deg2rad(c)), 0.0) ** 2  # noqa: E731
    amax = np.deg2rad(m.out * w(0) + m.up * w(90) + m.inn * w(180) + m.down * w(-90))
    a0 = np.deg2rad(m.a0)
    span = np.maximum(amax - a0, 1e-6)
    squeezed = a0 + span * np.tanh((th - a0) / span)
    th_new = np.where((th <= a0) | (amax >= np.deg2rad(179.0)), th, squeezed)
    perp = u.copy()
    perp[:, 0] = 0
    pn = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = np.where(pn > 1e-6, perp / np.maximum(pn, 1e-9), np.array([0, 0, -1.0]))
    return (np.cos(th_new)[:, None] * np.array([1.0, 0, 0]) + np.sin(th_new)[:, None] * perp) * n


def targets(kp: dict, side: str, ik: ArmIk, pitch: float, k: float, c: np.ndarray,
            a0: float = 180.0, amax: float = 180.0, mapping: Mapping | None = None):
    A = pitch_matrix(pitch)
    sh, el, wr = kp[f"{side}_shoulder"], kp[f"{side}_elbow"], kp[f"{side}_wrist"]
    squeeze = (lambda v: map_dirs(v, side, mapping)) if mapping else (lambda v: compress_cone(v, a0, amax))
    wv = squeeze((wr - sh) @ A.T)
    wt = ik.shoulder + c + k * wv
    u = squeeze((el - sh) @ A.T)
    et = ik.shoulder + L_UPPER * u / np.linalg.norm(u, axis=1, keepdims=True)
    f = (wr - el) @ A.T
    hd = f / np.linalg.norm(f, axis=1, keepdims=True)
    return wt, et, hd


_NEUTRAL: dict = {}


def _branch(arm):
    """Human-like branch: upper arm down-forward, elbow forward (J2 < 0), J1 mid-range.

    left base Y points down, right base Y points up, so 'upper arm down' is
    J1 = -pi/2 (left) / +pi/2 (right) with J2 < 0; mirror-symmetric postures.
    """
    s1 = -np.pi / 2 if arm == "left" else np.pi / 2
    return np.array([s1, -0.9, 0.0, -1.6, 0.0, 1.9, 0.0])


def neutral_posture(arm):
    """Comfortable reference: wrist 0.5 m forward, 0.15 m below J2, on the human-like branch."""
    if arm not in _NEUTRAL:
        ik = ArmIk(arm)
        q0 = _branch(arm)
        wt = ik.shoulder + np.array([0.50, 0.0, -0.15])
        et = ik.shoulder + L_UPPER * np.array([0.3, 0.0, -0.95])
        _NEUTRAL[arm] = ik.solve(q0, wt, et, np.array([1.0, 0, 0]), q0, iters=300)
    return _NEUTRAL[arm]


def _seeds(arm):
    b = _branch(arm)
    out = 1.0 if arm == "left" else -1.0   # J1 toward the outboard (sideways) solutions
    return [b + np.array([d1, d2, d3, 0, 0, 0, 0])
            for d1 in (-0.8, 0, 0.8, 1.6 * out, 2.4 * out) for d2 in (-0.5, 0.0, 0.4) for d3 in (-0.6, 0.6)]


def seed_solve(ik: ArmIk, wt, et, hd, q_prev=None, q_post=None, cont_w=0.05):
    """Best IK over several seeds (plus q_prev); cost = wrist error + cont_w*|q - q_prev|."""
    seeds = _seeds(ik.arm) + ([q_prev] if q_prev is not None else [])
    best, bq = 1e9, None
    for q0 in seeds:
        q = ik.solve(q0, wt, et, hd, q0 if q_post is None else q_post, iters=150)
        cost = np.linalg.norm(ik.fk(q)[0] - wt)
        if q_post is not None:
            cost += 0.02 * np.linalg.norm(q - q_post)
        if q_prev is not None:
            cost += cont_w * np.linalg.norm(q - q_prev)
        if cost < best:
            best, bq = cost, q
    return bq


def run_clip(ik: ArmIk, kp: dict, side: str, pitch, k, c, stride=1, clearance=True,
             a0=180.0, amax=180.0, restart_tol=None, obstacles=None, mapping: Mapping | None = None):
    """obstacles: optional per-output-frame point clouds (e.g. the other arm) to keep clear of.
    mapping: a Mapping (overrides pitch/k/a0/amax)."""
    if mapping is not None:
        pitch, k = mapping.pitch, mapping.k
    wt, et, hd = targets(kp, side, ik, pitch, k, np.asarray(c, float), a0, amax, mapping)
    idx = np.arange(0, len(wt), stride)
    q_post = neutral_posture(ik.arm)
    q = seed_solve(ik, wt[0], et[0], hd[0], q_post=q_post)
    Q, err, clr = [], [], []
    for n, i in enumerate(idx):
        obs = None if obstacles is None else obstacles[n]
        q_new = ik.solve(q, wt[i], et[i], hd[i], q_post, obstacle=obs)
        if restart_tol is not None and np.linalg.norm(ik.fk(q_new)[0] - wt[i]) > restart_tol:
            q_new = seed_solve(ik, wt[i], et[i], hd[i], q_prev=q, q_post=q_post)
        q = q_new
        Q.append(q)
        err.append(np.linalg.norm(ik.fk(q)[0] - wt[i]))
        clr.append(ik.clearance()[0] if clearance else np.nan)
    return np.array(Q), np.array(err), np.array(clr), idx


# --------------------------------------------------------------------------
# One pipeline for process and sweep (stage 5): both arms, left first, the right arm
# avoids the left arm; validity = clearance, inter-arm distance and (new) wrist error.
# --------------------------------------------------------------------------
SPEED_REF = 0.3        # target peak joint speed of the reference (0.6 x 0.5 rad/s cap)
MIN_SEG_S = 5.0


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
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


def retarget_clip(clip: str, m: Mapping, stride: int = 1) -> dict:
    """Retarget one clip with mapping m. Returns per-frame arrays (at source fps / stride)."""
    kp = arm_keypoints(BVH_DIR / f"{clip}.bvh")
    fps = kp["fps"] / stride
    iks = {s: ArmIk(s, weights=m.weights()) for s in ("left", "right")}
    Q, err = {}, {}
    q, e, _, idx = run_clip(iks["left"], kp, "left", 0, 0, (0, 0, 0), stride=stride, clearance=False, mapping=m)
    Q["left"], err["left"] = gaussian_smooth(q, 2.0 / stride), e
    clouds = []
    for t in range(len(Q["left"])):
        iks["left"].set_q(Q["left"][t])
        clouds.append(iks["left"].coarse_world_verts())
    q, e, _, _ = run_clip(iks["right"], kp, "right", 0, 0, (0, 0, 0), stride=stride, clearance=False,
                          obstacles=clouds, mapping=m)
    Q["right"], err["right"] = gaussian_smooth(q, 2.0 / stride), e
    L, R = iks["left"], iks["right"]
    d = L.d
    T = len(Q["left"])
    clr = {"left": np.zeros(T), "right": np.zeros(T)}
    who = {"left": [], "right": []}
    inter = np.zeros(T)
    for t in range(T):
        d.qpos[L.qadr] = Q["left"][t]
        d.qpos[R.qadr] = Q["right"][t]
        mujoco.mj_kinematics(L.m, d)
        for side, ik in (("left", L), ("right", R)):
            ik.d.qpos[:] = d.qpos
            mujoco.mj_kinematics(ik.m, ik.d)
            c, w = ik.env_clearance()
            clr[side][t] = c
            who[side].append(w)
        inter[t] = inter_arm_distance(L, R)
    valid = (clr["left"] >= CLEARANCE_MIN) & (clr["right"] >= CLEARANCE_MIN) & (inter >= CLEARANCE_MIN)
    if m.wrist_tol_valid is not None:
        valid &= (err["left"] <= m.wrist_tol_valid) & (err["right"] <= m.wrist_tol_valid)
    return dict(clip=clip, kp=kp, idx=idx, fps=fps, Q=np.concatenate([Q["left"], Q["right"]], axis=1),
                err=err, clr=clr, who=who, inter=inter, valid=valid)


def clip_summary(r: dict) -> dict:
    Qb, fps, valid, T = r["Q"], r["fps"], r["valid"], len(r["valid"])
    qd = np.gradient(Qb, axis=0) * fps
    segs = []
    for a, b in segments(valid, int(MIN_SEG_S * fps)):
        peak = float(np.abs(qd[a:b]).max())
        segs.append(dict(start=a, end=b, seconds=(b - a) / fps, peak_speed=peak,
                         p99_speed=float(np.percentile(np.abs(qd[a:b]).max(axis=1), 99)),
                         slowdown=max(1.0, peak / SPEED_REF)))
    fails = {}
    for side in ("left", "right"):
        bad = [w for w, c in zip(r["who"][side], r["clr"][side]) if c < CLEARANCE_MIN]
        for w in set(bad):
            fails[f"{side}:{w}"] = bad.count(w) / T
        fails[f"{side}:wrist_err"] = float(np.mean(r["err"][side] > (MAPPING_TOL or 9)))
    fails["inter_arm"] = float(np.mean(r["inter"] < CLEARANCE_MIN))
    return dict(frames=T, seconds=T / fps, valid=float(valid.mean()),
                track_left=float(np.mean(r["err"]["left"] < WRIST_TOL)),
                track_right=float(np.mean(r["err"]["right"] < WRIST_TOL)),
                fails=fails, segments=segs)


MAPPING_TOL = None   # wrist_tol_valid of the mapping being processed (for the fail report)


def _process_clip(job):
    clip, m = job
    global MAPPING_TOL
    MAPPING_TOL = m.wrist_tol_valid
    r = retarget_clip(clip, m, stride=1)
    out_dir = RET_DIR / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{clip}.npz", q=r["Q"], fps=r["fps"], valid=r["valid"],
                        clr_left=r["clr"]["left"], clr_right=r["clr"]["right"], inter=r["inter"],
                        err_left=r["err"]["left"], err_right=r["err"]["right"])
    return clip, clip_summary(r)


def process(args, m: Mapping):
    summary = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for clip, s in ex.map(_process_clip, [(c, m) for c in CANDIDATES]):
            summary[clip] = s
            seg_s = sum(x["seconds"] for x in s["segments"])
            print(f"{clip:28s} {s['seconds']:6.1f}s valid={s['valid']:.2f} segs={len(s['segments'])} "
                  f"seg_total={seg_s:6.1f}s fails={ {k: round(v, 2) for k, v in s['fails'].items() if v > 0.01} }", flush=True)
    (RET_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    (RET_DIR / "mapping.json").write_text(json.dumps(asdict(m), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", action="store_true", help="full-rate retarget of CANDIDATES")
    ap.add_argument("--mapping", type=Path, default=None,
                    help="JSON with Mapping fields (default: V1, the stage-4 cone)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--set", default=None, help="reference set to write (v1 is frozen)")
    ap.add_argument("--allow-v1", action="store_true")
    a = ap.parse_args()
    global RET_DIR
    if not (a.process and a.set):
        raise SystemExit("usage: --process --set <name> [--mapping m.json]   (the sweep is tools/retarget_sweep.py)")
    RET_DIR = panda_sets.get(a.set, write=True, allow_v1=a.allow_v1).ret
    RET_DIR.mkdir(parents=True, exist_ok=True)
    m = Mapping(**json.loads(a.mapping.read_text())) if a.mapping else V1
    print("mapping:", asdict(m))
    process(a, m)


if __name__ == "__main__":
    main()
