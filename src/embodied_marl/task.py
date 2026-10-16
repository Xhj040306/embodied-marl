"""协作推箱子任务的几何定义与奖励函数（纯函数，便于单元测试）。"""

from __future__ import annotations

import numpy as np

#: 目标区域中心（与 assets/coop_push.xml 中 goal 的位姿保持一致）
GOAL_XY = np.array([0.0, 0.95])

#: 达成判定的半径，单位：米
SUCCESS_RADIUS = 0.22


def push_distance(puck_xy: np.ndarray) -> float:
    """方块到目标中心的距离。"""
    return float(np.linalg.norm(np.asarray(puck_xy, dtype=float) - GOAL_XY))


def is_success(puck_xy: np.ndarray, radius: float = SUCCESS_RADIUS) -> bool:
    """方块是否已被推入目标区域。"""
    return push_distance(puck_xy) < radius


def shaping_reward(
    puck_xy: np.ndarray,
    prev_distance: float,
    agent_positions: np.ndarray,
    control: np.ndarray,
    *,
    control_cost: float = 0.01,
) -> tuple[float, float]:
    """稠密奖励：推动进度为主，控制代价为辅。

    Returns
    -------
    reward, distance
        本步奖励，以及本步之后方块到目标的距离（供下一步使用）。
    """
    distance = push_distance(puck_xy)
    progress = prev_distance - distance
    effort = control_cost * float(np.sum(np.square(control)))
    return progress - effort, distance
