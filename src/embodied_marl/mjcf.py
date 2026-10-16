r"""MJCF 模型加载工具（Windows 非 ASCII 路径适配）。

为什么需要这个模块
------------------
MuJoCo 底层是 C++，用窄字符（char*）打开文件。在 Windows 中文系统上，Python 传入的
UTF-8 路径会被按 ANSI 代码页（GBK）解释，于是文件打不开：

    >>> mujoco.MjModel.from_xml_path(r"C:\Users\someone\Desktop\作品集\scene.xml")
    ValueError: ParseXML: Error opening file '...\??Ʒ??\scene.xml'

实测结论（本仓库踩坑记录）
--------------------------
* `from_xml_path` 读非 ASCII 路径 -> 失败（如上）。
* `from_xml_string`（本模块采用）   -> 成功。Python 自己负责文件 IO，正确处理 Unicode。
* 第三方库自带的 MJCF 资源（例如 gymnasium 的内置 MuJoCo 环境）会从 site-packages 读取，
  所以如果**虚拟环境目录含非 ASCII 字符**，那些环境仍会失败（本项目不使用它们）。
* 本仓库只用自有 MJCF + 内置几何体，因此**标准虚拟环境即可**，无需特殊配置。

限制：这种方式不支持 <include> 与外部 mesh/texture 文件。本仓库模型只使用
MuJoCo 内置几何体（box/sphere/capsule/cylinder）与 builtin 纹理，因此不受影响。
"""

from __future__ import annotations

from pathlib import Path

import mujoco

__all__ = ["load_model", "load_model_from_text"]


def load_model_from_text(xml_text: str) -> mujoco.MjModel:
    """从 MJCF 字符串构建模型。"""
    return mujoco.MjModel.from_xml_string(xml_text)


def load_model(path: str | Path) -> mujoco.MjModel:
    """从任意（含非 ASCII）路径加载 MJCF 模型。

    Parameters
    ----------
    path:
        MJCF 文件路径，允许包含中文等非 ASCII 字符。

    Returns
    -------
    mujoco.MjModel
    """
    return load_model_from_text(Path(path).read_text(encoding="utf-8"))
