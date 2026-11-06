"""Render a retargeted dual-Panda reference (runs/panda_retarget/clips/<clip>.npz) to mp4.

    MUJOCO_GL=egl uv run python tools/render_panda_ref.py dance2_subject1 --start 20 --seconds 10
"""
import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
XML = REPO / "src" / "unisupermimic" / "assets" / "robots" / "panda" / "panda_left_right.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--slowdown", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    z = np.load(REPO / "runs" / "panda_retarget" / "clips" / f"{a.clip}.npz")
    q, fps, valid = z["q"], float(z["fps"]), z["valid"]
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 640)
    opt = mujoco.MjvOption()
    opt.geomgroup[5] = 1
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.35, 0, -0.1]
    cam.distance, cam.azimuth, cam.elevation = 2.4, 180, -12
    out = a.out or REPO / "runs" / "panda_retarget" / "videos" / f"{a.clip}_{a.start:g}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_fps = 30
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", "640x480", "-r", str(out_fps), "-i", "-", "-pix_fmt", "yuv420p", str(out)],
                          stdin=subprocess.PIPE)
    n_out = int(a.seconds * a.slowdown * out_fps)
    for i in range(n_out):
        t = a.start + i / out_fps / a.slowdown
        f = min(int(round(t * fps)), len(q) - 1)
        d.qpos[:] = q[f]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam, scene_option=opt)
        img = r.render()
        if not valid[f]:
            img[:12] = [220, 40, 40]   # red bar: frame fails a clearance check
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out)


if __name__ == "__main__":
    main()
