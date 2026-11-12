"""Manager-Based terms for the fixed-base Panda SuperMimic task.

Adapted from the G1 terms. Fixed-base differences:
  * state(): the root is the static 'mount' body, so (p, R, v, w) are constants
    (zeros / identity) and only joint state is read from the simulator;
  * reset writes joint state only (write_joint_state_to_sim_mask), with
    +-0.1 rad joint noise and no root noise;
  * the context xi holds per-joint encoder offsets only (no friction / CoM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import (
    ActionTerm,
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    EventTermCfg,
    ManagerRng,
    ManagerTermBase,
)

from unisupermimic.rl.contracts.supermimic import State

from . import robot as R
from .kinematics import FixedBaseKinematics
from .mdp import MAX_STEP_REWARD, XI_SCALE, FixedBaseMDP, RewardCfg
from .motion import BIN_FRAMES, MotionLib

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv as RuntimeEnv
    from unilab.managers.types import ManagerBasedRlEnv

RESET_JOINT_NOISE = 0.1


def start_bin_probabilities(bin_G, bin_w_avg, gamma: float) -> torch.Tensor:
    soft = torch.softmax(-25.0 * bin_G / (MAX_STEP_REWARD / (1.0 - gamma)), dim=0)
    bin_w_avg.mul_(0.999).add_(0.001 * soft)
    return 0.5 / bin_G.numel() + 0.4 * soft + 0.1 * bin_w_avg / bin_w_avg.sum()


def sample_start_frames(motion: MotionLib, bin_G, bin_w_avg, gamma: float, num_samples: int,
                        rng: ManagerRng) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = start_bin_probabilities(bin_G, bin_w_avg, gamma)
    bins = rng.choice(motion.num_bins, (num_samples,), p=probabilities)
    start = motion.bin_start[bins]
    end = torch.minimum(start + BIN_FRAMES, motion.clip_end[motion.bin_clip[bins]])
    frames = start + (rng.random((num_samples,)) * (end - start).float()).long()
    return frames, bins


def update_start_bin_returns(bin_G, start_bin, ep_step, ep_ret, ep_disc_ret, hard, done,
                             gamma: float) -> dict[str, torch.Tensor]:
    if not bool(done.any()):
        return {}
    lengths = ep_step[done].float()
    bootstrap = torch.where(hard[done], 0.0, gamma**lengths * ep_ret[done] / lengths / (1.0 - gamma))
    returns = ep_disc_ret[done] + bootstrap
    bins = start_bin[done]
    unique_bins, inverse, counts = torch.unique(bins, return_inverse=True, return_counts=True)
    summed = torch.zeros_like(unique_bins, dtype=returns.dtype).scatter_add_(0, inverse, returns)
    bin_G[unique_bins] = 0.8 * bin_G[unique_bins] + 0.2 * summed / counts
    return {"ep_len": lengths, "ep_ret_norm": ep_ret[done] / (lengths * MAX_STEP_REWARD)}


@dataclass(kw_only=True)
class PandaSuperMimicCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    variant: str = "left"
    motion_files: list[str]
    mjcf_path: str
    eval_mode: bool = False
    gamma: float = 0.99
    src_fps: float = 50.0
    reset_noise: bool = True
    max_episode_steps: int = 500
    goal_frames: int = 10
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    # JSON {"q0": {arm: [7]}, "alpha": {arm: [7]}} from tools/derive_q0_alpha.py for a reference
    # set other than v1; None = the v1 constants in robot.py.
    q0_alpha_file: str | None = None

    def build(self, env: ManagerBasedRlEnv) -> PandaSuperMimicCommand:
        return PandaSuperMimicCommand(self, env)


class PandaSuperMimicCommand(CommandTerm):
    """Reference motion, joint-offset context and episode bookkeeping (fixed base)."""

    cfg: PandaSuperMimicCommandCfg

    def __init__(self, cfg: PandaSuperMimicCommandCfg, env: ManagerBasedRlEnv):
        if cfg.variant not in R.ARMS:
            raise ValueError(f"variant must be one of {sorted(R.ARMS)}")
        if not 0.0 < cfg.gamma < 1.0 or not math.isfinite(cfg.gamma):
            raise ValueError("gamma must be in (0, 1)")
        super().__init__(cfg, env)
        self.robot = cast("Entity", env.scene[cfg.entity_name])
        names = R.joint_names(cfg.variant)
        if tuple(self.robot.joint_names) != tuple(names):
            raise ValueError(f"robot joint order {self.robot.joint_names} != {names}")
        device = self._device
        self.J = len(names)
        self.kin = FixedBaseKinematics(cfg.mjcf_path, device, names, R.tracked_bodies(cfg.variant), R.ROOT_BODY)
        self.motion = MotionLib(cfg.motion_files, self.kin, device)
        q0_tab, alpha_tab = R.load_q0_alpha(cfg.q0_alpha_file)
        q0 = torch.tensor(R.per_joint(q0_tab, cfg.variant), dtype=torch.float32, device=device)
        self.default_q = q0.expand(self.num_envs, -1).clone()
        self.alpha = torch.tensor(R.per_joint(alpha_tab, cfg.variant), dtype=torch.float32, device=device)
        self.mdp = FixedBaseMDP(self.kin, self.motion, q0.clone(), RewardCfg())
        self.gamma = float(cfg.gamma)
        self.eval_mode = cfg.eval_mode
        self.reset_noise = cfg.reset_noise and not cfg.eval_mode
        self.max_episode_steps = 10**9 if cfg.eval_mode else cfg.max_episode_steps
        n = self.num_envs
        self.f = torch.zeros(n, dtype=torch.int64, device=device)
        self.xi = torch.zeros((n, self.J), dtype=torch.float32, device=device)
        self.a_prev = torch.zeros((n, self.J), dtype=torch.float32, device=device)
        self.ep_step = torch.zeros(n, dtype=torch.int64, device=device)
        self.start_bin = torch.zeros(n, dtype=torch.int64, device=device)
        self.ep_disc_ret = torch.zeros(n, dtype=torch.float32, device=device)
        self.ep_ret = torch.zeros(n, dtype=torch.float32, device=device)
        self.bin_G = torch.zeros(self.motion.num_bins, dtype=torch.float32, device=device)
        self.bin_w_avg = torch.full((self.motion.num_bins,), 1.0 / self.motion.num_bins, device=device)
        self._root = (torch.zeros(n, 3, device=device), torch.eye(3, device=device).expand(n, 3, 3).clone(),
                      torch.zeros(n, 3, device=device), torch.zeros(n, 3, device=device))
        self.last_episode_stats: dict[str, torch.Tensor] = {}
        self._requested_frames: torch.Tensor | None = None
        self._requested_env_ids: torch.Tensor | None = None
        self._read_phase_bound = False

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (R.ROOT_BODY,)

    @property
    def command(self) -> torch.Tensor:
        return self.goal(self.cfg.goal_frames)

    def bind_read_phase(self) -> None:
        read_plan = self._env.scene.read_plan
        if read_plan is None:
            raise RuntimeError("SuperMimic requires the compiled scene tensor read plan")
        read_plan.refresh()
        self._read_phase_bound = True

    def state(self) -> State:
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None or not read_plan.ready:
            if self._read_phase_bound:
                raise RuntimeError("SuperMimic requires a refreshed scene tensor read phase")
            return self.mdp.ref_state(self.f)
        joint = read_plan.joint_tensor_view(self.robot)
        return (*self._root, joint.joint_pos, joint.joint_vel)

    def observation(self) -> torch.Tensor:
        return self.mdp.obs(self.state(), self.a_prev, self.xi, self.f)

    def goal(self, num_frames: int) -> torch.Tensor:
        return self.mdp.goal(self.f, num_frames)

    def request_frames(self, frames: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        expected = self.num_envs if env_ids is None else env_ids.numel()
        if frames.shape != (expected,):
            raise ValueError(f"reset frames must have shape ({expected},)")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self._device)
        else:
            env_ids = resolve_reset_env_ids(env_ids, num_envs=self.num_envs, device=self._device,
                                            owner="PandaSuperMimicCommand.request_frames")
        requested = frames.to(device=self._device, dtype=torch.int64).clone()
        if bool((requested < 0).any()) or bool((requested >= self.motion.num_frames).any()):
            raise ValueError("reset frames are outside the motion library")
        self._requested_frames, self._requested_env_ids = requested, env_ids

    def reset(self, env_ids: torch.Tensor | slice | None = None, *, publish_metrics: bool = True) -> dict[str, float]:
        del publish_metrics
        ids = resolve_reset_env_ids(env_ids, num_envs=self.num_envs, device=self._device,
                                    owner="PandaSuperMimicCommand.reset")
        reset_mask = self._reset_mask_buffer.resolve(ids, owner="PandaSuperMimicCommand.reset")
        if ids.numel() == 0:
            return {}
        if self._requested_frames is None:
            frames, bins = sample_start_frames(self.motion, self.bin_G, self.bin_w_avg, self.gamma,
                                               ids.numel(), self._rng)
        else:
            assert self._requested_env_ids is not None
            if not torch.equal(ids, self._requested_env_ids):
                raise ValueError("requested-frame reset environment IDs do not match the request")
            frames = self._requested_frames
            self._requested_frames = self._requested_env_ids = None
            bins = self.motion.frame_bin[frames]
        full_frames = self.f.clone()
        full_frames[ids] = frames
        ref = self.motion.ref_state(full_frames)
        q, qd = ref[4].clone(), ref[5].clone()
        if self.reset_noise:
            q[ids] += self._rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, (ids.numel(), self.J))
        q = torch.clamp(q, self.kin.q_lo + 0.02, self.kin.q_hi - 0.02)
        self.robot.write_joint_state_to_sim_mask(q, qd, reset_mask)
        self.f[ids] = frames
        self.ep_step[ids] = 0
        self.a_prev[ids] = 0.0
        self.start_bin[ids] = bins
        self.ep_disc_ret[ids] = 0.0
        self.ep_ret[ids] = 0.0
        self.time_left[ids] = self._resampling_time_range[0]
        self.command_counter[ids] += 1
        return {}

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        if bool(reset_mask.any()):
            raise RuntimeError("SuperMimic commands resample only through the reset lifecycle")

    def _update_command(self) -> None:
        reward = cast("RuntimeEnv", self._env).reward_buf
        self.ep_disc_ret.add_(self.gamma ** self.ep_step.float() * reward)
        self.ep_ret.add_(reward)
        self.f.add_(1)
        self.ep_step.add_(1)
        self.a_prev.copy_(self._env.action_manager.action)
        if self.eval_mode:
            self.last_episode_stats = {}
            return
        self.last_episode_stats = update_start_bin_returns(
            self.bin_G, self.start_bin, self.ep_step, self.ep_ret, self.ep_disc_ret,
            self._env.termination_manager.terminated, self._env.reset_buf, self.gamma)

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def post_compute_reset(self) -> None:
        return


@dataclass(kw_only=True)
class PandaSuperMimicActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    command_name: str = "motion"

    def build(self, env: ManagerBasedRlEnv) -> PandaSuperMimicAction:
        return PandaSuperMimicAction(self, env)


class PandaSuperMimicAction(ActionTerm):
    """Setpoint q_sp = q0 + xi + alpha * a for the real-robot-gain position actuators."""

    cfg: PandaSuperMimicActionCfg

    def __init__(self, cfg: PandaSuperMimicActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = _command(env, cfg.command_name)
        if self._command.robot is not self._entity:
            raise ValueError("action and command must control the same entity")
        J = self._command.J
        self._raw_actions = torch.zeros((self.num_envs, J), dtype=torch.float32, device=env.device)
        self._target = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self._command.J

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def tensor_target(self) -> torch.Tensor:
        return self._target

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)

    def apply_actions(self) -> None:
        c = self._command
        self._target.copy_(c.default_q + c.xi + c.alpha * self._raw_actions)
        cast("Entity", self._entity).set_full_width_joint_position_target(self._target)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = resolve_reset_env_ids(env_ids, num_envs=self.num_envs, device=self._env.device,
                                    owner="PandaSuperMimicAction.reset")
        self._raw_actions[ids] = 0.0


class PandaJointOffsetEvent(ManagerTermBase):
    """Sample per-episode encoder offsets xi ~ U(-0.01, 0.01) rad (zero in eval)."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        if cfg.mode != "reset":
            raise ValueError("PandaJointOffsetEvent requires reset mode")
        self._command_name = cfg.params.get("command_name", "motion")

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, command_name: str = "motion") -> None:
        del command_name
        ids = resolve_reset_env_ids(env_ids, num_envs=env.num_envs, device=env.device,
                                    owner="PandaJointOffsetEvent")
        command = _command(env, self._command_name)
        if command.eval_mode:
            command.xi[ids] = 0.0
        else:
            command.xi[ids] = env.rng.uniform(-XI_SCALE, XI_SCALE, (ids.numel(), command.J))


def supermimic_obs(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    return _command(env, command_name).observation()


def supermimic_goal(env: ManagerBasedRlEnv, command_name: str = "motion", num_frames: int = 1) -> torch.Tensor:
    return _command(env, command_name).goal(num_frames)


def supermimic_reward(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    c = _command(env, command_name)
    return c.mdp.reward(c.state(), c.f + 1, env.action_manager.action, c.a_prev)


def supermimic_termination(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    c = _command(env, command_name)
    return c.mdp.termination(c.state(), c.f + 1)


def supermimic_timeout(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    c = _command(env, command_name)
    return c.ep_step + 1 >= c.max_episode_steps


def supermimic_clip_end(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    c = _command(env, command_name)
    return c.f + 1 >= c.motion.clip_end[c.motion.clip_id[c.f]]


def _command(env: ManagerBasedRlEnv, command_name: str) -> PandaSuperMimicCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, PandaSuperMimicCommand):
        raise TypeError(f"Command term '{command_name}' is {type(command).__name__}")
    return command
