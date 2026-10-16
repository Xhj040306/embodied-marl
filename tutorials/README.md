# 从零实现的算法

这些是我自己实现的算法版本，用于确认对原理的理解，而不是直接调用现成实现。

| 文件 | 内容 |
| --- | --- |
| [`01_rl_basics.py`](01_rl_basics.py) | 表格型 Q-learning 解 FrozenLake，不依赖神经网络 |

## 运行

```bash
uv run python tutorials/01_rl_basics.py
```

输出学到的策略、贪心评估成功率，并生成学习曲线到 `docs/tutorial_qlearning.png`。
