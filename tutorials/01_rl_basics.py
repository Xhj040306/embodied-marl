"""表格型 Q-learning 求解 FrozenLake。

环境
    gymnasium 的 FrozenLake-v1：4x4 网格，16 个离散状态、4 个动作（左 / 下 / 右 / 上）。
    走到终点奖励 +1，掉进洞或超时奖励 0。

算法
    Q-learning，异策略时序差分，更新式：

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    不需要神经网络，Q 表本身就是全部参数。共 16 x 4 = 64 个待学习的值。

参考
    本脚本用于对照后续基于神经网络的策略梯度实现（见同目录后续文件）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只出图不开窗口，保证在无显示环境下也能运行

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from _plot_style import setup_cjk_font

CJK = setup_cjk_font()  # 返回 None 表示系统无中文字体，图形标签改用英文

ROOT = Path(__file__).resolve().parents[1]

N_STATES = 16
N_ACTIONS = 4
ARROWS = "←↓→↑"  # 与 gymnasium FrozenLake 的动作顺序一致


def epsilon_greedy(q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    """ε-贪心动作选择：以 ε 概率随机动作，否则取当前 Q 值最大的动作。

    Q 表初始全为 0，此时 argmax 恒返回第一个动作，智能体会原地撞墙。
    探索项用于打破这种退化，这是表格型方法能收敛的前提。
    """
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(q[state]))


def train(
    is_slippery: bool,
    episodes: int = 4000,
    alpha: float = 0.15,
    gamma: float = 0.97,
    eps_start: float = 1.0,
    eps_end: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """训练 Q 表，返回 (Q 表, 每回合是否成功)。

    is_slippery=True 时，执行的动作有 2/3 概率被环境随机改写为垂直方向，
    这决定了该环境下可达的成功率上限。
    """
    rng = np.random.default_rng(seed)
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)

    q = np.zeros((N_STATES, N_ACTIONS))
    success = np.zeros(episodes, dtype=bool)

    for episode in range(episodes):
        # 探索率线性衰减：前期以随机探索为主，后期逐步依赖已学到的最优动作
        epsilon = max(eps_end, eps_start + (eps_end - eps_start) * episode / episodes * 0.9)

        state, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done = False
        while not done:
            action = epsilon_greedy(q, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated and reward > 0:
                success[episode] = True

            # 时序差分更新。终止状态没有后续价值，future 取 0
            future = 0.0 if terminated else gamma * float(np.max(q[next_state]))
            q[state, action] += alpha * (reward + future - q[state, action])

            state = next_state

    env.close()
    return q, success


def evaluate(q: np.ndarray, is_slippery: bool, runs: int = 300, seed: int = 123) -> float:
    """用训练好的 Q 表做纯贪心评估，返回成功率。

    这才是策略的真实性能。训练过程中的成功率会被探索噪声拉低。
    """
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    wins = 0
    for i in range(runs):
        state, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q[state])))
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
    env.close()
    return wins / runs


def print_policy(q: np.ndarray, title: str) -> None:
    """把 Q 表还原成 4x4 的方向网格。"""
    print(f"\n{title}")
    for row in range(4):
        line = "  ".join(ARROWS[int(np.argmax(q[row * 4 + col]))] for col in range(4))
        print("   " + line)


def rolling_mean(values: np.ndarray, window: int = 100) -> np.ndarray:
    """滑动平均，把每回合的 0/1 成功标记转成可读的成功率曲线。"""
    pad = np.concatenate([np.zeros(window - 1, dtype=float), values.astype(float)])
    return np.convolve(pad, np.ones(window) / window, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-learning on FrozenLake")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "tutorial_qlearning.png")
    args = parser.parse_args()

    results = {}
    for slippery in (False, True):
        label = "is_slippery=True" if slippery else "is_slippery=False"
        print(f"\n=== 训练中：{label} ===")
        q, success = train(slippery, episodes=args.episodes)
        rolling = rolling_mean(success)
        rate = evaluate(q, slippery)
        results[slippery] = (rolling, rate)
        print(f"   最后 100 回合成功率: {success[-100:].mean():.1%}")
        print(f"   贪心评估成功率: {rate:.1%}")
        print_policy(q, f"学到的策略（{label}）：")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    for slippery, (rolling, rate) in results.items():
        if CJK:
            name = "is_slippery=True（动作打滑）" if slippery else "is_slippery=False（不滑）"
            suffix = f"贪心评估 {rate:.0%}"
        else:
            name = "is_slippery=True" if slippery else "is_slippery=False"
            suffix = f"greedy eval {rate:.0%}"
        ax.plot(rolling, linewidth=2, label=f"{name} · {suffix}")
    ax.set_xlabel("训练回合 Episode" if CJK else "Training episode")
    ax.set_ylabel(
        "成功率（最近 100 回合滑动平均）" if CJK else "Success rate (100-episode moving avg)"
    )
    ax.set_title("Q-learning 在 FrozenLake 上的学习曲线" if CJK else "Q-learning on FrozenLake")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    note = (
        "训练过程中保留探索，故曲线终点低于贪心评估成功率"
        if CJK
        else "Training curves include exploration noise"
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.03, note, ha="center", fontsize=8.5, alpha=0.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"\n中文字体: {CJK or '未找到，图形标签使用英文'}")
    print(f"学习曲线已保存: {args.out}")


if __name__ == "__main__":
    main()
