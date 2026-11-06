"""Minimal BVH reader for LAFAN1 and extraction of torso-relative arm keypoints.

LAFAN1 BVH: Y-up, centimetres, 30 fps; every joint has 3 rotation channels in
Z-Y-X order (the root also has 3 position channels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Bvh:
    names: list[str]
    parents: np.ndarray       # (J,)
    offsets: np.ndarray       # (J, 3) cm
    channels: list[list[str]]
    motion: np.ndarray        # (T, C)
    frame_time: float

    def index(self, name: str) -> int:
        return self.names.index(name)


def read_bvh(path: str | Path) -> Bvh:
    text = Path(path).read_text().split("MOTION")
    header, motion = text[0], text[1]
    names, parents, offsets, channels = [], [], [], []
    stack: list[int] = []
    last = -1
    for line in header.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] in ("ROOT", "JOINT"):
            names.append(tok[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            last = len(names) - 1
        elif tok[0] == "End":
            last = -2  # end site: ignore its offset
        elif tok[0] == "{":
            stack.append(last)
        elif tok[0] == "}":
            stack.pop()
        elif tok[0] == "OFFSET" and last >= 0:
            offsets[last] = np.array([float(v) for v in tok[1:4]])
        elif tok[0] == "CHANNELS":
            channels[last] = tok[2:]
    lines = motion.strip().splitlines()
    ft = float(lines[1].split(":")[1])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[2:] if ln.strip()])
    return Bvh(names, np.array(parents), np.array(offsets), channels, data, ft)


def _rot(axis: str, ang: np.ndarray) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    o, z = np.ones_like(ang), np.zeros_like(ang)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def global_positions(b: Bvh) -> np.ndarray:
    """(T, J, 3) joint positions in metres, BVH world frame (Y-up)."""
    T, J = len(b.motion), len(b.names)
    pos = np.zeros((T, J, 3))
    rot = np.zeros((T, J, 3, 3))
    col = 0
    for j in range(J):
        R = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        trans = None
        for ch in b.channels[j]:
            v = b.motion[:, col]
            col += 1
            if ch.endswith("position"):
                if trans is None:
                    trans = np.zeros((T, 3))
                trans[:, "XYZ".index(ch[0])] = v
            else:
                R = R @ _rot(ch[0], np.deg2rad(v))
        p = b.parents[j]
        if p < 0:
            pos[:, j] = trans if trans is not None else b.offsets[j]
            rot[:, j] = R
        else:
            pos[:, j] = pos[:, p] + np.einsum("tij,j->ti", rot[:, p], b.offsets[j])
            rot[:, j] = rot[:, p] @ R
    return pos / 100.0


# BVH world (Y-up) -> right-handed Z-up frame used below: (x, y, z) = (X, -Z, Y)
_Y_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def arm_keypoints(path: str | Path) -> dict:
    """Shoulder/elbow/wrist of both arms expressed in a torso frame.

    Torso frame: +x forward, +y the actor's left, +z up (along the spine),
    origin at the mid-point between the two shoulder joints.
    Returns arrays of shape (T, 3) in metres plus 'fps'.
    """
    b = read_bvh(path)
    P = global_positions(b) @ _Y_TO_Z_UP.T
    g = {n: P[:, b.index(n)] for n in ("Spine", "Neck", "LeftArm", "RightArm",
                                         "LeftForeArm", "RightForeArm", "LeftHand", "RightHand")}
    left = g["LeftArm"] - g["RightArm"]
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    up = g["Neck"] - g["Spine"]
    up -= np.sum(up * left, axis=1, keepdims=True) * left
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    fwd = np.cross(left, up)
    R = np.stack([fwd, left, up], axis=2)             # columns = torso axes in world
    origin = 0.5 * (g["LeftArm"] + g["RightArm"])

    def to_torso(p):
        return np.einsum("tji,tj->ti", R, p - origin)

    out = {"fps": 1.0 / b.frame_time}
    for side in ("Left", "Right"):
        k = side.lower()
        out[f"{k}_shoulder"] = to_torso(g[f"{side}Arm"])
        out[f"{k}_elbow"] = to_torso(g[f"{side}ForeArm"])
        out[f"{k}_wrist"] = to_torso(g[f"{side}Hand"])
    return out
