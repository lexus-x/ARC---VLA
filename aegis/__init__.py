"""
AEGIS-VLA: Adaptive Energy-Guided Interaction Synthesis
for Vision-Language-Action Models.

A novel VLA action-generation module that replaces standard regression
or diffusion-based action heads with:
  1. An Interaction Energy Network (IEN) that scores candidate actions
  2. Criticality-Aware Adaptive Computation (CAAC) for dynamic refinement
  3. Geometric Interaction Tokens (GIT) for explicit spatial grounding
"""

from aegis.energy_head import InteractionEnergyNetwork
from aegis.criticality import CriticalityEstimator
from aegis.geometric_tokens import GeometricInteractionTokens
from aegis.aegis_module import AEGISActionHead, AEGISConfig
from aegis.losses import aegis_loss

__version__ = "0.1.0"
__all__ = [
    "InteractionEnergyNetwork",
    "CriticalityEstimator",
    "GeometricInteractionTokens",
    "AEGISActionHead",
    "AEGISConfig",
    "aegis_loss",
]
