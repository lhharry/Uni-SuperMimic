"""Compact-pack motion library for fixed-base arms (any joint count J).

Row layout: [root_p 3, root_quat_wxyz 4, q J, root_v 3, root_w 3, qd J] at 50 Hz,
as written by tools/build_panda_motions.py.
"""

from __future__ import annotations

import copy

import torch

from unisupermimic.rl.algos.supermimic.geometry import mat_to_6d, matmul, matvec, quat_to_mat, yaw_mat
from unisupermimic.rl.contracts.supermimic import State
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import (
    BIN_FRAMES,
    CONTROL_DT,
    _load_compact_payload,
    compact_motion_clip_count,
)

from .kinematics import FixedBaseKinematics

__all__ = ["BIN_FRAMES", "CONTROL_DT", "MotionLib", "compact_motion_clip_count"]


class MotionLib:
    def __init__(self, files, kin: FixedBaseKinematics, device, src_fps: float = 50.0):
        del src_fps
        if len(files) != 1 or not str(files[0]).endswith(".pt"):
            raise ValueError("Panda MotionLib expects exactly one compact .pt pack")
        self.device = torch.device(device)
        self.kin = kin
        self.compact = True
        J = kin.num_joints
        self.J = J
        pack = _load_compact_payload(files[0])
        ref = pack["ref"]
        if ref.ndim != 2 or ref.shape[1] != 13 + 2 * J:
            raise ValueError(f"{files[0]}: ref must be (N, {13 + 2 * J}), got {tuple(ref.shape)}")
        expected = 0
        for start, length in zip(pack["starts"], pack["lengths"]):
            if start != expected or length < 2:
                raise ValueError(f"{files[0]}: invalid clip ranges")
            expected += length
        if expected != ref.shape[0]:
            raise ValueError(f"{files[0]}: clip lengths do not match ref frames")
        self._ref = ref.float().to(self.device)
        self.names = list(pack["names"])
        self.clip_start = torch.tensor(pack["starts"], dtype=torch.long, device=self.device)
        self.clip_len = torch.tensor(pack["lengths"], dtype=torch.long, device=self.device)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = expected
        self._sl = dict(q=slice(7, 7 + J), v=slice(7 + J, 10 + J), w=slice(10 + J, 13 + J),
                        qd=slice(13 + J, 13 + 2 * J))
        self._build_indices()

    def _build_indices(self) -> None:
        clips = torch.arange(len(self.names), dtype=torch.long, device=self.device)
        self.clip_id = torch.repeat_interleave(clips, self.clip_len)
        bins_per_clip = (self.clip_len + BIN_FRAMES - 1) // BIN_FRAMES
        self.clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.bin_clip = torch.repeat_interleave(clips, bins_per_clip)
        local_bin = torch.arange(len(self.bin_clip), device=self.device) - self.clip_bin_offset[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + local_bin * BIN_FRAMES
        self.num_bins = len(self.bin_clip)
        frames = torch.arange(self.num_frames, device=self.device)
        self.frame_bin = self.clip_bin_offset[self.clip_id] + (frames - self.clip_start[self.clip_id]) // BIN_FRAMES

    @property
    def q(self) -> torch.Tensor:
        return self._ref[:, self._sl["q"]]

    @property
    def qd(self) -> torch.Tensor:
        return self._ref[:, self._sl["qd"]]

    def ref_state(self, frames: torch.Tensor) -> State:
        r = self._ref[frames]
        return (r[..., :3], quat_to_mat(r[..., 3:7]), r[..., self._sl["v"]], r[..., self._sl["w"]],
                r[..., self._sl["q"]], r[..., self._sl["qd"]])

    def body_state(self, frames: torch.Tensor):
        return self.kin.forward(*self.ref_state(frames))

    def to(self, device) -> MotionLib:
        device = torch.device(device)
        if device == self.device:
            return self
        new = copy.copy(self)
        new.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, name, value.to(device))
        new.kin = self.kin.to(device)
        return new

    def future_frames(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        end = self.clip_end[self.clip_id[frames]]
        future = frames.unsqueeze(-1) + torch.arange(1, num_frames + 1, device=frames.device)
        return torch.minimum(future, end.unsqueeze(-1))

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Future reference features, same layout as G1 (root terms are constant here)."""
        future = self.future_frames(frames, num_frames)
        r = self._ref[future]
        current_rotation = quat_to_mat(self._ref[frames, 3:7])
        root_R = quat_to_mat(r[..., 3:7])
        heading_t = yaw_mat(current_rotation).transpose(-1, -2).unsqueeze(1)
        position_delta = matvec(heading_t, r[..., :3] - self._ref[frames, :3].unsqueeze(1))
        rotation = mat_to_6d(matmul(heading_t, root_R))
        root_v_local = matvec(root_R.transpose(-1, -2), r[..., self._sl["v"]])
        root_w_local = matvec(root_R.transpose(-1, -2), r[..., self._sl["w"]])
        features = torch.cat((position_delta, rotation, root_v_local, root_w_local,
                              r[..., self._sl["q"]], r[..., self._sl["qd"]] * 0.1), -1)
        return features.reshape(frames.shape[0], -1)
