"""Export everything the robot side (~/harry/franka_mimic) needs, without torch there.

  policy  : EMA policy weights as numpy (enc/dec MLPs, running norms) + q0, alpha,
            hist, goal_frames; verified against the torch Policy on random inputs.
  refs    : selected retimed segments with a smooth start/stop (time scaling ramps
            to zero speed over --ease seconds at both ends), q / qd for both arms at
            50 Hz, and the policy goal features for the left (J=7) and dual (J=14)
            variants, computed with the task's own MotionLib.goal.
  model   : the torque-motor dual-Panda MJCF + meshes (franka_mimic simulates
            libfranka's joint impedance itself).

    uv run python tools/export_deploy.py --ckpt-left runs/panda_left_001/ckpt_006000.pt \
        --ckpt-dual runs/panda_dual_001/ckpt_006000.pt --out ~/harry/franka_mimic
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panda_arm_ik import ArmIk, inter_arm_distance  # noqa: E402

from unisupermimic.rl.algos.supermimic import networks  # noqa: E402
from unisupermimic.tasks.motion_tracking.panda.supermimic import robot as R  # noqa: E402
from unisupermimic.tasks.motion_tracking.panda.supermimic.kinematics import FixedBaseKinematics  # noqa: E402
from unisupermimic.tasks.motion_tracking.panda.supermimic.motion import MotionLib  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
import panda_sets  # noqa: E402

DT = 0.02


def export_policy(ckpt_path: Path, variant: str, out: Path, q0_alpha_file=None) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    q0_tab, alpha_tab = R.load_q0_alpha(str(q0_alpha_file) if q0_alpha_file else None)
    J = len(R.joint_names(variant))
    obs_dim, goal_dim = 3 * J, (15 + 2 * J) * int(args["goal_frames"])
    networks.AMP_DTYPE = None
    pol = networks.Policy(obs_dim, goal_dim, hist=int(args["hist"]), z_dim=int(args["z_dim"]),
                          num_joints=J, hidden=tuple(args["policy_hidden"]))
    pol.load_state_dict(ck["pi"])
    pol.eval()
    arrays = {}
    for name, seq in (("enc", pol.enc), ("dec", pol.dec)):
        lin = [m for m in seq if isinstance(m, torch.nn.Linear)]
        for i, m in enumerate(lin):
            arrays[f"{name}_W{i}"] = m.weight.detach().numpy().astype(np.float32)
            arrays[f"{name}_b{i}"] = m.bias.detach().numpy().astype(np.float32)
        arrays[f"{name}_n"] = np.array(len(lin))
    for name, norm in (("obs", pol.obs_norm), ("goal", pol.goal_norm)):
        arrays[f"{name}_mean"] = norm.mean.numpy().astype(np.float32)
        arrays[f"{name}_std"] = norm.std().numpy().astype(np.float32)
    meta = dict(variant=variant, hist=int(args["hist"]), goal_frames=int(args["goal_frames"]),
                q0=R.per_joint(q0_tab, variant), alpha=R.per_joint(alpha_tab, variant),
                joint_names=R.joint_names(variant), kp=R.per_joint(R.KP, variant), kd=R.per_joint(R.KD, variant),
                ckpt=str(ckpt_path), iteration=int(ck.get("iteration", -1)))
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(out, **arrays)

    # verify numpy forward == torch forward
    def elu(x):
        return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))

    def mlp(prefix, x):
        n = int(arrays[f"{prefix}_n"])
        for i in range(n):
            x = x @ arrays[f"{prefix}_W{i}"].T + arrays[f"{prefix}_b{i}"]
            if i < n - 1:
                x = elu(x)
        return x

    rng = np.random.default_rng(0)
    oh = rng.normal(size=(8, meta["hist"], obs_dim)).astype(np.float32)
    g = rng.normal(size=(8, goal_dim)).astype(np.float32)
    z = np.tanh(mlp("enc", (g - arrays["goal_mean"]) / arrays["goal_std"]))
    o = ((oh - arrays["obs_mean"]) / arrays["obs_std"]).reshape(8, -1)
    a_np = mlp("dec", np.concatenate([z, o], -1))
    with torch.no_grad():
        a_t = pol(torch.tensor(oh), torch.tensor(g)).numpy()
    err = np.abs(a_np - a_t).max()
    print(f"policy {variant}: {ckpt_path} -> {out}  numpy-vs-torch max diff {err:.2e}")
    assert err < 1e-4, err


def ease(q: np.ndarray, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-scale so the speed ramps 0 -> 1 -> 0 (raised cosine) over `seconds` at both ends."""
    T = len(q)
    t_src = np.arange(T) * DT
    n_ramp = int(seconds / DT)
    # output time grid: integrate the scaling s(tau) until the source time is exhausted
    taus, ts = [0.0], [0.0]
    while ts[-1] < t_src[-1]:
        tau = taus[-1]
        s_in = 0.5 - 0.5 * np.cos(np.pi * min(tau / seconds, 1.0))
        s_out = 0.5 - 0.5 * np.cos(np.pi * min((t_src[-1] - ts[-1]) / (0.5 * seconds), 1.0))
        s = max(min(s_in, s_out), 0.02)
        taus.append(tau + DT)
        ts.append(min(ts[-1] + s * DT, t_src[-1]))
    ts = np.array(ts)
    qe = np.stack([np.interp(ts, t_src, q[:, j]) for j in range(q.shape[1])], 1)
    # hold the end pose for a moment so the final velocity is exactly zero
    qe = np.concatenate([qe, np.repeat(qe[-1:], n_ramp // 2, 0)])
    qde = np.gradient(qe, DT, axis=0)
    qde[0] = qde[-1] = 0.0
    return qe, qde


def goal_features(q: np.ndarray, qd: np.ndarray, variant: str, frames: int = 10) -> np.ndarray:
    """MotionLib.goal on a one-clip pack built from q/qd (constant root)."""
    J = len(R.joint_names(variant))
    sl = slice(0, J)
    n = len(q)
    root = np.zeros((n, 7), np.float32)
    root[:, 3] = 1.0
    ref = np.concatenate([root, q[:, sl], np.zeros((n, 6)), qd[:, sl]], 1).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "seg.pt"
        torch.save({"ref": torch.tensor(ref), "names": ["seg"], "starts": [0], "lengths": [n]}, p)
        kin = FixedBaseKinematics(R.MJCF[variant], "cpu", R.joint_names(variant), R.tracked_bodies(variant))
        lib = MotionLib([str(p)], kin, "cpu")
        return lib.goal(torch.arange(n), frames).numpy()


def clearances(q: np.ndarray) -> tuple[float, float]:
    L, Rr = ArmIk("left"), ArmIk("right")
    env, inter = 1.0, 1.0
    for t in range(0, len(q), 5):
        L.set_q(q[t, :7])
        Rr.set_q(q[t, 7:])
        env = min(env, L.env_clearance()[0], Rr.env_clearance()[0])
        inter = min(inter, inter_arm_distance(L, Rr))
    return env, inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-left", type=Path, required=True)
    ap.add_argument("--ckpt-dual", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-refs", type=int, default=12)
    ap.add_argument("--min-s", type=float, default=8.0, help="min retimed segment length")
    ap.add_argument("--max-s", type=float, default=40.0)
    ap.add_argument("--ease", type=float, default=1.5)
    panda_sets.add_args(ap, write=True)
    a = ap.parse_args()
    out = a.out.expanduser()
    rs = panda_sets.get(a.set, write=True, allow_v1=a.allow_v1)
    pol_dir, ref_dir = out / "policy" / rs.name, out / "refs" / rs.name     # franka_mimic layout per set
    pol_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(parents=True, exist_ok=True)

    export_policy(a.ckpt_left, "left", pol_dir / "left.npz", rs.q0alpha)
    export_policy(a.ckpt_dual, "dual", pol_dir / "dual.npz", rs.q0alpha)

    src = REPO / "src" / "unisupermimic" / "assets" / "robots" / "panda"
    shutil.copy(src / "panda_left_right.xml", out / "model" / "panda_left_right.xml")
    if not (out / "model" / "assets").exists():
        shutil.copytree(src / "assets", out / "model" / "assets")

    pack = torch.load(rs.pack("dual"), weights_only=True)
    ref = pack["ref"].numpy()
    cands = []
    for name, s, n in zip(pack["names"], pack["starts"], pack["lengths"]):
        if not (a.min_s <= n * DT <= a.max_s):
            continue
        q = ref[s:s + n, 7:21].astype(np.float64)
        env, inter = clearances(q)
        cands.append((min(env, inter), name, q, env, inter))
    cands.sort(key=lambda c: -c[0])
    index = []
    for score, name, q, env, inter in cands[: a.n_refs]:
        qe, qde = ease(q, a.ease)
        env_e, inter_e = clearances(qe)
        np.savez(ref_dir / f"{name}.npz", q=qe.astype(np.float32), qd=qde.astype(np.float32), dt=DT,
                 goal_left=goal_features(qe, qde, "left"), goal_dual=goal_features(qe, qde, "dual"))
        row = dict(name=name, seconds=round(len(qe) * DT, 1), min_env_clearance=round(env_e, 3),
                   min_inter_arm=round(inter_e, 3), peak_speed=round(float(np.abs(qde).max()), 3),
                   left_j1_range=round(float(np.ptp(qe[:, 0])), 2))
        index.append(row)
        print(row)
    (ref_dir / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
