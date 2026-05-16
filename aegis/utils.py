"""Utilities for AEGIS-VLA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    """Human-readable parameter count."""
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    return f"{n / 1_000_000:.1f}M"


def cosine_schedule(
    step: int,
    total_steps: int,
    start: float,
    end: float,
) -> float:
    """Cosine annealing schedule."""
    if step >= total_steps:
        return end
    progress = step / total_steps
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def make_negative_samples(
    gt_action: torch.Tensor,
    num_samples: int,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """
    Generate negative action samples by perturbing GT actions.

    Args:
        gt_action:   (B, A)
        num_samples: K  number of negatives per batch element.
        noise_scale: std of Gaussian perturbation.

    Returns:
        negatives: (B, K, A)
    """
    B, A = gt_action.shape
    noise = torch.randn(B, num_samples, A, device=gt_action.device) * noise_scale
    return gt_action.unsqueeze(1) + noise


class EMAModel:
    """Exponential moving average of model parameters for stable inference."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}
