"""
Differentiable Top-K selection via Gumbel-Softmax.

Enables gradient flow through discrete token selection during training.
At inference, uses straight top-K (no noise).
"""

import torch
import torch.nn.functional as F


def gumbel_topk(
    weights: torch.Tensor,
    k: int,
    tau: float = 1.0,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable top-K selection.
    
    During training: Gumbel-Softmax with temperature annealing.
    At inference: Straight top-K (deterministic).
    
    Args:
        weights: (batch, N) importance weights in (0, 1).
        k: Number of tokens to select.
        tau: Gumbel-Softmax temperature (higher = more exploration).
        training: Whether in training mode.
    
    Returns:
        selected_indices: (batch, k) indices of selected tokens.
        soft_weights: (batch, N) soft selection weights (for gradient flow).
    """
    if not training:
        # Deterministic top-K at inference
        selected_indices = torch.topk(weights, k, dim=-1).indices
        soft_weights = torch.zeros_like(weights)
        soft_weights.scatter_(1, selected_indices, 1.0)
        return selected_indices, soft_weights
    
    # Training: Gumbel-Softmax
    # Add Gumbel noise to log-weights
    log_weights = torch.log(weights + 1e-8)
    u = torch.rand_like(log_weights)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    noisy_logits = (log_weights + gumbel_noise) / tau
    
    # Softmax to get soft weights
    soft_weights = F.softmax(noisy_logits, dim=-1)
    
    # Hard top-K for indexing (straight-through: gradients flow through soft_weights)
    selected_indices = torch.topk(soft_weights, k, dim=-1).indices
    
    return selected_indices, soft_weights


def select_tokens(
    visual_tokens: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Gather selected visual tokens by index.
    
    Args:
        visual_tokens: (batch, N, d) full visual token sequence.
        selected_indices: (batch, k) indices to select.
    
    Returns:
        selected: (batch, k, d) selected visual tokens.
    """
    # Expand indices for gathering
    indices_expanded = selected_indices.unsqueeze(-1).expand(
        -1, -1, visual_tokens.shape[-1]
    )
    return torch.gather(visual_tokens, 1, indices_expanded)
