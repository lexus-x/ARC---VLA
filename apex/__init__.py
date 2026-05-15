"""
APEX: Adaptive Pruning and Extraction for Vision-Language-Action Models.

A plug-and-play module that learns to select task-relevant visual tokens,
making any VLA faster AND more accurate with ~33K trainable parameters.
"""

from apex.router import APEXRouter
from apex.wrapper import APEXWrappedVLA, wrap_vla
from apex.gumbel_topk import gumbel_topk
from apex.losses import apex_loss

__version__ = "0.1.0"
__all__ = [
    "APEXRouter",
    "APEXWrappedVLA",
    "wrap_vla",
    "gumbel_topk",
    "apex_loss",
]
