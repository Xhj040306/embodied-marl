import numpy as np

from embodied_marl.task import GOAL_XY, is_success, push_distance, shaping_reward


def test_push_distance_zero_at_goal():
    assert push_distance(GOAL_XY) == 0.0


def test_is_success_boundary():
    assert is_success(GOAL_XY)
    assert not is_success(GOAL_XY + np.array([0.5, 0.0]))


def test_shaping_reward_rewards_progress():
    control = np.zeros(4)
    reward, distance = shaping_reward(
        GOAL_XY + np.array([0.4, 0.0]), 0.5, np.zeros((2, 2)), control
    )
    assert reward > 0  # 靠近目标应得到正奖励
    assert distance < 0.5


def test_control_cost_is_penalised():
    puck = GOAL_XY + np.array([0.4, 0.0])
    assert shaping_reward(puck, 0.4, np.zeros((2, 2)), np.zeros(4))[0] == 0.0
    assert shaping_reward(puck, 0.4, np.zeros((2, 2)), np.ones(4))[0] < 0.0
