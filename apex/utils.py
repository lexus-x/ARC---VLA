"""Utility functions for APEX."""

import torch
import math


def cosine_anneal(step: int, total_steps: int, start: float = 1.0, end: float = 0.1) -> float:
    """Cosine annealing for Gumbel temperature."""
    if step >= total_steps:
        return end
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * step / total_steps))


def compute_spatial_positions(
    num_tokens: int,
    image_size: int = 224,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Compute 2D spatial positions for ViT patch tokens.
    
    Args:
        num_tokens: Number of visual tokens (N).
        image_size: Input image size.
        patch_size: ViT patch size.
    
    Returns:
        positions: (N, 2) normalized 2D positions in [0, 1].
    """
    grid_size = image_size // patch_size
    assert grid_size * grid_size == num_tokens, (
        f"num_tokens ({num_tokens}) != grid_size^2 ({grid_size}^2)"
    )
    
    # Create grid
    y = torch.linspace(0, 1, grid_size)
    x = torch.linspace(0, 1, grid_size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    # Flatten to (N, 2)
    positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    return positions


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n < 1_000:
        return str(n)
    elif n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    else:
        return f"{n / 1_000_000:.1f}M"
