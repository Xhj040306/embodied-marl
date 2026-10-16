"""运行一段仿真，可选导出为 GIF。

用法：
    uv run python scripts/rollout.py                # 生成 docs/rollout.gif
    uv run python scripts/rollout.py --no-render    # 只跑物理，用于 CI
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from embodied_marl import load_model

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "coop_push.xml"
OUT_DEFAULT = ROOT / "docs" / "rollout.gif"


def build_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.0, 0.45, 0.05]
    cam.distance = 2.6
    cam.azimuth = 145.0
    cam.elevation = -32.0
    return cam


def run(steps: int, render: bool, out_path: Path, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    model = load_model(SCENE)
    data = mujoco.MjData(model)

    renderer = None
    cam = None
    frames: list[np.ndarray] = []
    if render:
        renderer = mujoco.Renderer(model, height=432, width=576)
        cam = build_camera(model)

    # 随机策略，每 40 步重采一次动作，避免逐帧白噪声导致的抖动。
    control = np.zeros(model.nu)
    for step in range(steps):
        if step % 40 == 0:
            control = rng.uniform(-1.0, 1.0, size=model.nu).astype(np.float64)
        data.ctrl[:] = control
        mujoco.mj_step(model, data)

        if renderer is not None and step % 2 == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    puck = data.body("puck").xpos
    result = {
        "steps": steps,
        "sim_time": float(data.time),
        "puck_xy": (float(puck[0]), float(puck[1])),
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "nu": model.nu,
    }

    if renderer is not None:
        import imageio.v3 as iio

        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out_path, frames, duration=0.04, loop=0)
        result["frames"] = len(frames)
        result["gif"] = str(out_path)
        renderer.close()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Coop-push rollout smoke test")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true", help="skip rendering (for CI)")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    info = run(args.steps, render=not args.no_render, out_path=args.out, seed=args.seed)
    for key, value in info.items():
        print(f"{key:>10}: {value}")
    print("ROLLOUT: OK")


if __name__ == "__main__":
    main()
