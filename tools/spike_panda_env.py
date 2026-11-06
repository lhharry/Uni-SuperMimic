"""Stage 3.1 spike: fixed-base Panda SuperMimic env on mjwarp (4 envs). Usage: uv run python tools/spike_panda_env.py [task]"""
import sys
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path
import unisupermimic
from unilab.base.config_adapter import create_env
from unilab.training import ensure_registries
from unisupermimic.scripts.train_supermimic import build_env_override

task = sys.argv[1] if len(sys.argv) > 1 else "panda_super_mimic/mjwarp"
n = 4
cfg_dir = Path(unisupermimic.__file__).resolve().parent / "conf" / "supermimic"
with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
    cfg = compose(config_name="config", overrides=[f"task={task}", f"algo.num_envs={n}"])
ensure_registries()
env = create_env(cfg, num_envs=n, env_cfg_override=build_env_override(cfg, learner_device="cuda:0"))
cmd = env.supermimic_mdp
c = env._supermimic_command
print("J", c.J, "obs_dim", cmd.obs_dim, "goal_frame_dim", cmd.goal_frame_dim, "frames", c.motion.num_frames)
obs, info = env.reset()
print("obs keys", {k: tuple(v.shape) for k, v in obs.items()})
s = c.state(); print("state shapes", [tuple(x.shape) for x in s])
f0 = c.f.clone()
ref_q = c.motion.ref_state(c.f)[4]
print("reset q - ref q (noise expected <=0.1):", (s[4] - ref_q).abs().max().item())

# 1) open-loop 'perfect' actions: a = (q_ref(f+1) - q0 - xi) / alpha
errs, rews, terms = [], [], 0
for t in range(150):
    qn = c.motion.ref_state(c.f + 1)[4]
    a = (qn - c.default_q - c.xi) / c.alpha
    st = env.step(a); rew, term, info = st.reward, st.terminated, st.info
    s = c.state()
    errs.append((s[4] - c.motion.ref_state(c.f)[4]).abs().max().item())
    rews.append(rew.mean().item())
    terms += int(term.sum())
print(f"feed-forward tracking: max|q-q_ref| last50={max(errs[-50:]):.4f} rad, mean reward/step={sum(rews)/len(rews):.4f} (max {cmd.max_step_reward:.3f}), hard terms={terms}")
tr = info.get("supermimic")
print("transition keys", sorted(tr.keys()) if tr else None, "s dim", tr["s"].shape if tr else None)

# 2) gravcomp hold: request reset to frame, then action that holds the reset pose
env.reset_to_frames(f0)
q_hold = c.state()[4].clone()
for t in range(100):
    a = (q_hold - c.default_q - c.xi) / c.alpha
    env.step(a)
print("hold drift after 2 s (rad):", (c.state()[4] - q_hold).abs().max().item(), "(PD steady-state, gravcomp on)")
env.close(); env.backend.close()
print("SPIKE OK")
