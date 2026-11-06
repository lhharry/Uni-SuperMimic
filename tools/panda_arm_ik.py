"""Damped least-squares IK and clearance checks for one arm of the dual-Panda MJCF.

Task: the Panda wrist centre (link6 origin, where J5/J6 axes meet) follows a
wrist target, the elbow (link4 origin) follows an elbow hint, and the flange z
axis loosely follows the forearm direction ("straight hand"). Joint limits are
enforced with a margin by clipping; continuity comes from warm starting and a
small step-regularisation term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DUAL_XML = REPO / "src" / "unisupermimic" / "assets" / "robots" / "panda" / "panda_left_right.xml"

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
ENV_GEOMS = ("panel", "table", "keepout_side_left", "keepout_side_right", "keepout_head")


@dataclass
class IkWeights:
    wrist: float = 1.0
    elbow: float = 0.3
    hand_dir: float = 0.0
    step: float = 0.02
    posture: float = 0.005
    damping: float = 1e-3
    limit: float = 1.0          # soft joint-limit barrier
    collision: float = 2.0      # soft clearance barrier (panel, table, side walls)
    soft_margin: float = 0.45   # barrier starts this far from a joint limit
    safe_dist: float = 0.17     # barrier starts below this clearance (m)


@dataclass
class ArmIk:
    arm: str                       # "left" | "right"
    margin: float = 0.25
    weights: IkWeights = field(default_factory=IkWeights)
    xml: Path = DUAL_XML

    def __post_init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(self.xml))
        self.d = mujoco.MjData(self.m)
        j0 = self.m.joint(f"{self.arm}/joint1")
        self.qadr = np.arange(j0.qposadr[0], j0.qposadr[0] + 7)
        self.dadr = np.arange(j0.dofadr[0], j0.dofadr[0] + 7)
        self.lo, self.hi = Q_MIN + self.margin, Q_MAX - self.margin
        b = lambda n: self.m.body(f"{self.arm}/{n}").id
        self.b_elbow, self.b_wrist, self.b_flange = b("link4"), b("link6"), b("link7")
        mujoco.mj_forward(self.m, self.d)
        self.shoulder = self.d.xpos[b("link2")].copy()   # J2 centre, fixed
        self.base_R = self.d.xmat[b("link0")].reshape(3, 3).copy()
        skip = (b("link0"), b("link1"))
        self.arm_geoms = [g for g in range(self.m.ngeom)
                          if self.m.geom_contype[g] and self.m.geom_bodyid[g] not in skip
                          and self.m.body(self.m.geom_bodyid[g]).name.startswith(f"{self.arm}/")]
        self.env_geoms = [self.m.geom(n).id for n in ENV_GEOMS]
        self._verts = [self._local_verts(g) for g in self.arm_geoms]
        self._geom_body = np.array([self.m.geom_bodyid[g] for g in self.arm_geoms])
        self._vgeom = np.concatenate([np.full(len(v), i) for i, v in enumerate(self._verts)])
        m = self.m
        pid, tid, sid = m.geom("panel").id, m.geom("table").id, m.geom("keepout_side_left").id
        # planes as (normal, offset): clearance = n.p - off
        self._planes = [(np.array([1.0, 0, 0]), m.geom_pos[pid][0] + m.geom_size[pid][0]),
                        (np.array([0, 0, 1.0]), m.geom_pos[tid][2] + m.geom_size[tid][2]),
                        (np.array([0, -1.0, 0]), -(m.geom_pos[sid][1] - m.geom_size[sid][1])),
                        (np.array([0, 1.0, 0]), -(m.geom_pos[sid][1] - m.geom_size[sid][1]))]
        hb = m.geom("keepout_head").id
        self._head_lo, self._head_hi = m.geom_pos[hb] - m.geom_size[hb], m.geom_pos[hb] + m.geom_size[hb]
        self._jp = np.zeros((3, self.m.nv))
        self._jr = np.zeros((3, self.m.nv))

    # -- kinematics ---------------------------------------------------------
    def set_q(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def _point_jac(self, body):
        mujoco.mj_jacBody(self.m, self.d, self._jp, self._jr, body)
        return self._jp[:, self.dadr].copy(), self._jr[:, self.dadr].copy()

    def fk(self, q):
        self.set_q(q)
        R = self.d.xmat[self.b_flange].reshape(3, 3)
        return (self.d.xpos[self.b_wrist].copy(), self.d.xpos[self.b_elbow].copy(), R[:, 2].copy())

    # -- IK -----------------------------------------------------------------
    def solve(self, q_init, wrist_t, elbow_t, hand_dir, q_post, iters=30, tol=1e-4):
        w = self.weights
        q = np.clip(q_init.copy(), self.lo, self.hi)
        q_prev = q.copy()
        for _ in range(iters):
            self.set_q(q)
            pw, pe = self.d.xpos[self.b_wrist], self.d.xpos[self.b_elbow]
            z = self.d.xmat[self.b_flange].reshape(3, 3)[:, 2]
            Jw, _ = self._point_jac(self.b_wrist)
            Je, _ = self._point_jac(self.b_elbow)
            _, Jr = self._point_jac(self.b_flange)
            Jz = -np.cross(z[None, :], Jr.T).T          # d z / dq = jr x z
            # soft joint limits
            slo, shi = Q_MIN + w.soft_margin, Q_MAX - w.soft_margin
            rl = np.where(q > shi, q - shi, np.where(q < slo, q - slo, 0.0))
            Jl = np.diag((rl != 0).astype(float))
            rc, Jc = self._collision_rows()
            r = np.concatenate([w.wrist * (pw - wrist_t), w.elbow * (pe - elbow_t),
                                w.hand_dir * (z - hand_dir), w.step * (q - q_prev),
                                w.posture * (q - q_post), w.limit * rl, w.collision * rc])
            J = np.vstack([w.wrist * Jw, w.elbow * Je, w.hand_dir * Jz,
                           w.step * np.eye(7), w.posture * np.eye(7), w.limit * Jl,
                           w.collision * Jc])
            dq = -np.linalg.solve(J.T @ J + w.damping * np.eye(7), J.T @ r)
            q = np.clip(q + dq, self.lo, self.hi)
            if np.abs(dq).max() < tol:
                break
        self.set_q(q)
        return q

    def _collision_rows(self):
        """Hinge residuals (d - safe) < 0 for the closest vertex of each body to each plane."""
        V = self.world_verts()
        safe = self.weights.safe_dist
        rows, jacs = [], []
        for n, off in self._planes:
            dist = V @ n - off
            close = dist < safe
            if not close.any():
                continue
            for body in np.unique(self._geom_body[self._vgeom[close]]):
                sel = close & (self._geom_body[self._vgeom] == body)
                i = np.flatnonzero(sel)[np.argmin(dist[sel])]
                mujoco.mj_jac(self.m, self.d, self._jp, self._jr, V[i], int(body))
                rows.append(dist[i] - safe)
                jacs.append(n @ self._jp[:, self.dadr])
        # head keep-out box
        cp = np.clip(V, self._head_lo, self._head_hi)
        dv = V - cp
        dist = np.linalg.norm(dv, axis=1)
        close = dist < safe
        for body in (np.unique(self._geom_body[self._vgeom[close]]) if close.any() else []):
            sel = close & (self._geom_body[self._vgeom] == body)
            i = np.flatnonzero(sel)[np.argmin(dist[sel])]
            if dist[i] < 1e-6:
                continue
            mujoco.mj_jac(self.m, self.d, self._jp, self._jr, V[i], int(body))
            rows.append(dist[i] - safe)
            jacs.append((dv[i] / dist[i]) @ self._jp[:, self.dadr])
        if not rows:
            return np.zeros(0), np.zeros((0, 7))
        return np.array(rows), np.array(jacs)

    # -- clearance ----------------------------------------------------------
    def _local_verts(self, g):
        m = self.m
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            return v[:: max(1, len(v) // 400)].copy()
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            sz = m.geom_size[g]
            return np.array([[a * sz[0], b * sz[1], c * sz[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        r = m.geom_rbound[g]
        return np.array([[a * r, b * r, c * r] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])

    def world_verts(self):
        out = []
        for g, v in zip(self.arm_geoms, self._verts):
            out.append(v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g])
        return np.concatenate(out)

    def env_clearance(self, q=None):
        """Min distance (m) of this arm to panel, table, side keep-outs and head keep-out.

        Uses the collision-mesh vertices: exact for the planes (convex hull), a close
        approximation for the head box. mj_geomDistance is not used here because its
        mesh-vs-large-thin-box results were unreliable (spurious zeros).
        """
        if q is not None:
            self.set_q(q)
        V = self.world_verts()
        m = self.m
        panel_x = m.geom_pos[m.geom("panel").id][0] + m.geom_size[m.geom("panel").id][0]
        tbl = m.geom("table").id
        table_z = m.geom_pos[tbl][2] + m.geom_size[tbl][2]
        side_y = m.geom_pos[m.geom("keepout_side_left").id][1] - m.geom_size[m.geom("keepout_side_left").id][1]
        hb = m.geom("keepout_head").id
        lo = m.geom_pos[hb] - m.geom_size[hb]
        hi = m.geom_pos[hb] + m.geom_size[hb]
        dbox = np.linalg.norm(np.maximum(0, np.maximum(lo - V, V - hi)), axis=1)
        d = {"panel": (V[:, 0] - panel_x).min(), "table": (V[:, 2] - table_z).min(),
             "side": (side_y - np.abs(V[:, 1])).min(), "head": dbox.min()}
        k = min(d, key=d.get)
        return d[k], k

    def clearance(self, q=None, others=(), distmax=0.5):
        """Min distance (m) to the environment and to geoms `others` (e.g. the other arm)."""
        best, who = self.env_clearance(q)
        ft = np.zeros(6)
        for g1 in self.arm_geoms:
            for g2 in others:
                dist = mujoco.mj_geomDistance(self.m, self.d, g1, g2, distmax, ft)
                if dist < best:
                    best, who = dist, (self.m.geom(g1).name or self.m.body(self.m.geom_bodyid[g1]).name,
                                       self.m.geom(g2).name or self.m.body(self.m.geom_bodyid[g2]).name)
        return best, who


def inter_arm_distance(a: ArmIk, b: ArmIk) -> float:
    """Min vertex-to-vertex distance (m) between two arms at their current states.

    mj_geomDistance mesh-vs-mesh occasionally returns spurious zeros (seen in
    replay), so collision-mesh vertices are compared directly. Over-estimates
    the true distance by at most about half the vertex spacing (~1 cm).
    """
    va = a.world_verts().astype(np.float32)
    vb = b.world_verts().astype(np.float32)
    d2 = (va * va).sum(1)[:, None] + (vb * vb).sum(1)[None, :] - 2.0 * va @ vb.T
    return float(np.sqrt(max(d2.min(), 0.0)))
