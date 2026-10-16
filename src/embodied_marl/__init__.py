"""Embodied MARL: cooperative multi-agent tasks in MuJoCo."""

from .mjcf import load_model, load_model_from_text

__all__ = ["load_model", "load_model_from_text"]
__version__ = "0.1.0"
