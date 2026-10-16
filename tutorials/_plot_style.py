"""matplotlib 中文显示辅助。

matplotlib 默认字体（DejaVu Sans）不含中文字形，直接画中文会变成"豆腐块"方框。
这里自动在系统已安装字体中挑一个可用的中文字体；若一个都没有（例如精简的
Linux CI 环境），返回 None，调用方就改用英文标签，保证脚本在任何环境都不会崩。
"""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager

#: 按优先级排列的常见中文字体，覆盖 Windows / macOS / 常见 Linux 发行版
CJK_CANDIDATES = (
    "Microsoft YaHei",  # Windows 微软雅黑
    "SimHei",  # Windows 黑体
    "PingFang SC",  # macOS
    "Hiragino Sans GB",  # macOS
    "Noto Sans CJK SC",  # Linux
    "Source Han Sans SC",  # Linux
    "WenQuanYi Zen Hei",  # Linux
)


def setup_cjk_font() -> str | None:
    """挑选可用的中文字体并应用。返回字体名；没有可用字体时返回 None。"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
            return name
    return None
