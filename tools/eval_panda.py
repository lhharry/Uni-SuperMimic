"""Deterministic full-library evaluation of a Panda SuperMimic policy vs the PD baseline.

Every segment is played once from its first frame (eval mode: no noise, xi = 0)
in the same mjwarp task env with the same real-robot-gain actuators, by
  policy : the trained SuperMimic policy
  pd     : setpoint = next reference frame (q_sp = q_ref(f+1)), i.e. plain joint
           impedance tracking with dq_d = 0, which is what the robot would do
           without a policy.
  pd_ff  : the same plus velocity feed-forward (dq_d = qd_ref), realised here as
           q_sp = q_ref + (D/K) qd_ref; panda-py can send dq_d directly.
Reported: normalized return, failures (hard terminations), and elbow / wrist
position error per arm (RMSE and max, mm), weighted by frames.

    uv run python tools/eval_panda.py --ckpt runs/panda_left_001/last.pt [--task panda_super_mimic/mjwarp]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import panda_sets
import torch
from hydra import compose, initialize_config_dir
from unilab.base.config_adapter import create_env
from unilab.training import ensure_registries

import unisupermimic
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.scripts.train_supermimic import build_env_override


def make_env(task: str, num_envs: int, overrides=()):
    cfg_dir = Path(unisupermimic.__file__).resolve().parent / "conf" / "supermimic"
    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        cfg = compose(config_name="config", overrides=[f"task={task}", *overrides])
    ensure_registries()
    return create_env(cfg, num_envs=num_envs,
                      env_cfg_override=build_env_override(cfg, eval_mode=True, learner_device="cuda:0"))


def load_policy(mdp, ckpt: dict, device):
    args = ckpt["args"]
    networks.AMP_DTYPE = torch.bfloat16 if args.get("amp", "none") == "bf16" else None
    policy = networks.Policy(mdp.obs_dim, mdp.goal_frame_dim * int(args["goal_frames"]),
                             hist=int(args["hist"]), z_dim=int(args["z_dim"]), num_joints=mdp.num_joints,
                             hidden=tuple(args.get("test_hidden_dims") or args["policy_hidden"])).to(device)
    policy.load_state_dict(ckpt["pi"])
    policy.eval()
    return policy, int(args["goal_frames"]), int(args["hist"])


@torch.inference_mode()
def run(env, controller: str, policy=None, goal_frames=10, hist=10) -> dict:
    command = env._supermimic_command
    mdp, motion = command.mdp, command.motion
    tracked = [command.kin.body_names[i] for i in command.kin.tracked_idx]
    n_clips, n_env = len(motion.names), int(env.num_envs)
    order = motion.clip_len.argsort(descending=True, stable=True)
    starts = motion.clip_start[order].clone()
    _, info = env.reset_to_frames(starts[:n_env].clone())
    cur = info[SUPERMIMIC_STATE_KEY]
    dev = env.device
    ret = torch.zeros(n_clips, device=dev)
    steps = torch.zeros(n_clips, dtype=torch.int64, device=dev)
    failed = torch.zeros(n_clips, dtype=torch.bool, device=dev)
    sq = torch.zeros(n_clips, len(tracked), device=dev)
    mx = torch.zeros(n_clips, len(tracked), device=dev)
    assigned = order[:n_env].to(dev)
    nxt = n_env
    history = None
    from unisupermimic.tasks.motion_tracking.panda.supermimic import robot as R
    kd_over_kp = torch.tensor([d / k for d, k in zip(R.per_joint(R.KD, command.cfg.variant),
                                                       R.per_joint(R.KP, command.cfg.variant))], device=env.device)
    while bool((assigned >= 0).any()):
        if controller == "policy":
            obs = mdp.obs(unflatten_state(cur["s"]), cur["a_prev"], cur["xi"], cur["f"])
            history = obs.unsqueeze(1).repeat(1, hist, 1) if history is None else torch.cat(
                (history[:, 1:], obs.unsqueeze(1)), 1)
            action = policy(history, mdp.goal(cur["f"], goal_frames))
        else:
            ref_next = motion.ref_state(cur["f"] + 1)
            q_sp = ref_next[4]
            if controller == "pd_ff":
                # tau = K(q_sp - q) - D qd == K(q_ref - q) + D(qd_ref - qd) with this setpoint
                q_sp = q_sp + kd_over_kp * ref_next[5]
            action = (q_sp - command.default_q - cur["xi"]) / command.alpha
        st = env.step(action.to(dev))
        tr = st.info[SUPERMIMIC_TRANSITION_KEY]
        active_ids = (assigned >= 0).nonzero(as_tuple=False).flatten()
        clips = assigned[active_ids]
        ret[clips] += tr["r"][active_ids]
        steps[clips] += 1
        failed[clips] |= tr["term"][active_ids]
        s1 = unflatten_state(tr["s1"][active_ids])
        bp = command.kin.forward(*s1)[0]
        rp = motion.body_state(tr["f"][active_ids] + 1)[0]
        e = (bp - rp).norm(dim=-1)
        sq[clips] += e ** 2
        mx[clips] = torch.maximum(mx[clips], e)
        cur = st.info[SUPERMIMIC_STATE_KEY]
        done = (tr["done"] & (assigned >= 0)).nonzero(as_tuple=False).flatten()
        k = min(done.numel(), n_clips - nxt)
        if k:
            ids = done[:k]
            _, info = env.reset_to_frames(starts[nxt:nxt + k], env_ids=ids)
            cur = info[SUPERMIMIC_STATE_KEY]
            if history is not None:
                obs = mdp.obs(unflatten_state(cur["s"][ids]), cur["a_prev"][ids], cur["xi"][ids], cur["f"][ids])
                history[ids] = obs.unsqueeze(1).expand(-1, hist, -1)
            assigned[ids] = order[nxt:nxt + k].to(dev)
            nxt += k
        if k < done.numel():
            assigned[done[k:]] = -1
    max_ret = (motion.clip_len - 1).float() * mdp.max_step_reward
    w = steps.float()
    rmse = torch.sqrt(sq.sum(0) / w.sum())
    out = {
        "norm_return": (ret / max_ret).mean().item(),
        "norm_return_framew": (ret.sum() / max_ret.sum()).item(),
        "n_failed": int(failed.sum()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "rmse_mm": {b: 1000 * rmse[i].item() for i, b in enumerate(tracked)},
        "max_mm": {b: 1000 * mx[:, i].max().item() for i, b in enumerate(tracked)},
        "seconds": w.sum().item() * 0.02,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--task", default="panda_super_mimic/mjwarp")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--controllers", nargs="+", default=["policy", "pd", "pd_ff"])
    panda_sets.add_args(ap)      # which reference set to evaluate on (cross-evaluation: any ckpt x any set)
    a = ap.parse_args()
    ov = panda_sets.get(a.set).overrides("dual" if "dual" in a.task else "left")
    ckpt = torch.load(a.ckpt, map_location="cuda:0", weights_only=False)
    from unisupermimic.tasks.motion_tracking.panda.supermimic.motion import (
        compact_motion_clip_count,
    )
    probe_cfg_dir = Path(unisupermimic.__file__).resolve().parent / "conf" / "supermimic"
    with initialize_config_dir(version_base="1.3", config_dir=str(probe_cfg_dir)):
        cfg = compose(config_name="config", overrides=[f"task={a.task}", *ov])
    n = compact_motion_clip_count(str(cfg.env.commands.motion.motion_files[0]))
    env = make_env(a.task, n, ov)
    mdp = env._supermimic_command.mdp
    policy, gf, hist = load_policy(mdp, ckpt, env.device)
    result = {"ckpt": str(a.ckpt), "iteration": int(ckpt.get("iteration", -1)), "set": a.set}
    for ctrl in a.controllers:
        result[ctrl] = run(env, ctrl, policy, gf, hist)
    env.close()
    env.backend.close()
    txt = json.dumps(result, indent=1)
    print(txt)
    out = a.out or a.ckpt.parent / (f"eval_{a.ckpt.stem}.json" if a.set == "v1" else f"eval_{a.ckpt.stem}_{a.set}.json")
    out.write_text(txt)


if __name__ == "__main__":
    main()
