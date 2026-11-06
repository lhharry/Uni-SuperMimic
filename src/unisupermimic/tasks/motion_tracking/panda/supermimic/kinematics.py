"""Torch forward kinematics for fixed-base arms hanging off a static root body."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    axis_angle_to_mat,
    matmul,
    matvec,
    quat_to_mat,
)


class FixedBaseKinematics:
    """Same interface as G1Kinematics; index 0 is the static root (state p, R).

    Jointed bodies whose parent chain up to the root has no joints (e.g. link1
    under mount -> link0) get that fixed chain folded into their offset, computed
    once from the model at qpos0 relative to the root body.
    """

    def __init__(
        self,
        mjcf_path: str,
        device: str | torch.device,
        joint_names: Sequence[str],
        tracked: Sequence[str],
        root_body: str = "mount",
    ):
        import mujoco

        model = mujoco.MjModel.from_xml_path(mjcf_path)  # pyright: ignore[reportAttributeAccessIssue]
        data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_kinematics(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        self.device = torch.device(device)
        jidx = {name: i for i, name in enumerate(joint_names)}
        root_id = model.body(root_body).id
        R_root = data.xmat[root_id].reshape(3, 3)
        p_root = data.xpos[root_id]

        bodies, parent, joint_idx, axes, off_p, off_R = [root_body], [-1], [-1], [[0.0, 0.0, 1.0]], [[0.0] * 3], []
        off_R.append(torch.eye(3))
        local = {root_id: 0}
        for body_id in range(1, model.nbody):
            if model.body_jntnum[body_id] == 0:
                continue
            joint_id = model.body_jntadr[body_id]
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:  # pyright: ignore[reportAttributeAccessIssue]
                raise ValueError(f"{model.body(body_id).name}: only hinge joints are supported")
            name = model.joint(joint_id).name
            if name not in jidx:
                continue
            parent_id = int(model.body_parentid[body_id])
            if parent_id in local:
                p_idx = local[parent_id]
                op = model.body_pos[body_id]
                oq = torch.tensor(model.body_quat[body_id], dtype=torch.float32)
                oR = quat_to_mat(oq)
            else:
                # fixed chain: express the parent's world pose relative to the root
                p_idx = 0
                Rp = data.xmat[parent_id].reshape(3, 3)
                pp = data.xpos[parent_id]
                Rrel = R_root.T @ Rp
                prel = R_root.T @ (pp - p_root)
                op = prel + Rrel @ model.body_pos[body_id]
                oR = torch.tensor(Rrel, dtype=torch.float32) @ quat_to_mat(
                    torch.tensor(model.body_quat[body_id], dtype=torch.float32))
            local[body_id] = len(bodies)
            bodies.append(model.body(body_id).name)
            parent.append(p_idx)
            joint_idx.append(jidx[name])
            axes.append(model.jnt_axis[joint_id].tolist())
            off_p.append(list(map(float, op)))
            off_R.append(oR)
        missing = set(joint_names) - {model.joint(int(model.body_jntadr[b])).name for b in local if b != root_id}
        if missing:
            raise ValueError(f"joints not found under {root_body}: {sorted(missing)}")
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(axes, dtype=torch.float32, device=self.device)
        self.off_p = torch.tensor(off_p, dtype=torch.float32, device=self.device)
        self.off_R = torch.stack(off_R).to(self.device)
        self.tracked_idx = [bodies.index(name) for name in tracked]
        self.num_bodies = len(bodies)
        self.num_joints = len(joint_names)
        lo, hi = [0.0] * len(joint_names), [0.0] * len(joint_names)
        for name, i in jidx.items():
            lo[i], hi[i] = map(float, model.joint(name).range)
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=self.device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=self.device)

    def to(self, device: str | torch.device) -> FixedBaseKinematics:
        device = torch.device(device)
        if device == self.device:
            return self
        new = copy.copy(self)
        new.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, name, value.to(device))
        return new

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world poses and link-origin linear/angular velocities."""
        positions, rotations, lin, ang = [p], [R], [v], [w]
        for index in range(1, self.num_bodies):
            pi, ji = self.parent[index], self.joint_idx[index]
            parent_rotation = rotations[pi]
            position = positions[pi] + matvec(parent_rotation, self.off_p[index])
            offset_rotation = matmul(parent_rotation, self.off_R[index])
            rotation = matmul(offset_rotation, axis_angle_to_mat(self.axes[index], q[..., ji]))
            axis_world = matvec(offset_rotation, self.axes[index])
            angular = ang[pi] + axis_world * qd[..., ji : ji + 1]
            linear = lin[pi] + torch.cross(ang[pi], position - positions[pi], dim=-1)
            positions.append(position)
            rotations.append(rotation)
            lin.append(linear)
            ang.append(angular)
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([positions[i] for i in idx], -2),
            torch.stack([rotations[i] for i in idx], -3),
            torch.stack([lin[i] for i in idx], -2),
            torch.stack([ang[i] for i in idx], -2),
        )
