"""Differentiable SuperMimic heads for fixed-base arms.

Differences from G1 (see the port plan, stage 3.2):
  obs     = [q - q0 - xi, qd, a_prev]                         (3J)
  reward  = elbow/wrist position, orientation, linear and angular velocity
            + joint position, all exp-kernels; sigmas sized for 0.3 rad/s arm motion
  term    = any tracked body further than term_dist from its reference
  context = xi = per-joint encoder offsets only (J)
  fixed_base = True: the world model passes the (constant) root through.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from unisupermimic.rl.algos.supermimic.geometry import invariant_features, rot_angle_sq
from unisupermimic.rl.contracts.supermimic import State

from .kinematics import FixedBaseKinematics
from .motion import CONTROL_DT, MotionLib

NUM_TRACK_TERMS = 5
MAX_STEP_REWARD = NUM_TRACK_TERMS * 1.0 * CONTROL_DT
XI_SCALE = 0.01


@dataclass
class RewardCfg:
    body_pos_std: float = 0.07
    body_ori_std: float = 0.3
    body_linvel_std: float = 0.3
    body_angvel_std: float = 1.0
    joint_pos_std: float = 0.1
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_dist: float = 0.25
    term_temperature: float = 0.02


class FixedBaseMDP:
    fixed_base = True
    max_step_reward = MAX_STEP_REWARD

    def __init__(self, kin: FixedBaseKinematics, motion: MotionLib, default_q: torch.Tensor,
                 cfg: RewardCfg | None = None):
        J = kin.num_joints
        if default_q.shape != (J,):
            raise ValueError(f"default_q must have shape {(J,)}, got {tuple(default_q.shape)}")
        self.kin, self.motion = kin, motion
        self.cfg = RewardCfg() if cfg is None else cfg
        self.num_joints = J
        self.context_dim = J
        self.obs_dim = 3 * J
        self.goal_frame_dim = 15 + 2 * J
        self.default_q = default_q
        middle, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo = middle - half * self.cfg.soft_limit_factor
        self.soft_hi = middle + half * self.cfg.soft_limit_factor
        dev = default_q.device
        self.obs_noise = torch.cat((torch.full((J,), 0.01), torch.full((J,), 0.05), torch.zeros(J))).to(dev)
        self.obs_scale = torch.ones(3 * J, device=dev)

    def to(self, device) -> FixedBaseMDP:
        device = torch.device(device)
        if device == self.default_q.device:
            return self
        new = copy.copy(self)
        new.kin = self.kin.to(device)
        new.motion = self.motion.to(device)
        for name in ("default_q", "soft_lo", "soft_hi", "obs_noise", "obs_scale"):
            setattr(new, name, getattr(self, name).to(device))
        return new

    def obs(self, state: State, previous_action, context, frames) -> torch.Tensor:
        del frames
        joint_pos, joint_vel = state[4], state[5]
        return torch.cat((joint_pos - self.default_q - context, joint_vel, previous_action), -1)

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

    def goal(self, frames, num_frames: int) -> torch.Tensor:
        return self.motion.goal(frames, num_frames)

    def _errors(self, state: State, frames):
        body_p, body_R, body_v, body_w = self.kin.forward(*state)
        rp, rR, rv, rw = self.motion.body_state(frames)
        return (body_p, body_R, body_v, body_w), (rp, rR, rv, rw)

    def reward(self, state: State, frames, action, previous_action, terms: bool = False):
        cfg = self.cfg
        (bp, bR, bv, bw), (rp, rR, rv, rw) = self._errors(state, frames)
        e_pos = ((rp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(rR, bR).mean(-1)
        e_lin = ((rv - bv) ** 2).sum(-1).mean(-1)
        e_ang = ((rw - bw) ** 2).sum(-1).mean(-1)
        e_q = ((self.motion.ref_state(frames)[4] - state[4]) ** 2).mean(-1)
        tracking = (torch.exp(-e_pos / cfg.body_pos_std ** 2) + torch.exp(-e_ori / cfg.body_ori_std ** 2)
                    + torch.exp(-e_lin / cfg.body_linvel_std ** 2) + torch.exp(-e_ang / cfg.body_angvel_std ** 2)
                    + torch.exp(-e_q / cfg.joint_pos_std ** 2))
        action_rate = ((action - previous_action) ** 2).sum(-1)
        q = state[4]
        joint_limit = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        reward = (tracking + cfg.w_action_rate * action_rate + cfg.w_joint_limit * joint_limit) * CONTROL_DT
        if terms:
            return reward, {"track": tracking * CONTROL_DT, "e_body_pos": e_pos, "e_body_ori": e_ori,
                            "e_joint": e_q, "action_rate": action_rate, "joint_limit": joint_limit}
        return reward

    def _max_body_dist(self, state: State, frames) -> torch.Tensor:
        bp = self.kin.forward(*state)[0]
        rp = self.motion.body_state(frames)[0]
        return (rp - bp).norm(dim=-1).max(-1).values

    def termination(self, state: State, frames) -> torch.Tensor:
        return self._max_body_dist(state, frames) > self.cfg.term_dist

    def soft_termination(self, state: State, frames) -> torch.Tensor:
        d = self._max_body_dist(state, frames)
        return torch.sigmoid((d - self.cfg.term_dist) / self.cfg.term_temperature)

    def critic_features(self, state: State, frames, normalized_context) -> torch.Tensor:
        bp = self.kin.forward(*state)[0]
        rp = self.motion.body_state(frames)[0]
        ref_q = self.motion.ref_state(frames)[4]
        return torch.cat((invariant_features(state), ref_q - state[4], (rp - bp).flatten(-2),
                          normalized_context), -1)

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return context / XI_SCALE

    def valid_imagination_start(self, frames, horizon: int) -> torch.Tensor:
        return frames + horizon <= self.motion.clip_end[self.motion.clip_id[frames]]

    def ref_state(self, frames) -> State:
        return self.motion.ref_state(frames)

    def clip_info(self, frames) -> dict[str, torch.Tensor]:
        clips = self.motion.clip_id[frames]
        return {"clip_id": clips, "clip_start": self.motion.clip_start[clips],
                "clip_end": self.motion.clip_end[clips], "frame_bin": self.motion.frame_bin[frames]}
