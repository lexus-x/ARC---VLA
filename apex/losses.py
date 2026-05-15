"""
APEX training losses.

Three components:
1. L_action: MSE between predicted and ground-truth actions
2. L_coverage: Ensures selected tokens cover task-relevant spatial regions
3. L_diversity: Prevents selection of only spatially adjacent tokens
"""

import torch
import torch.nn as nn


def action_loss(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Action prediction MSE loss.
    
    Args:
        predicted: (batch, 6) predicted action from VLA.
        ground_truth: (batch, 6) ground-truth action.
    
    Returns:
        Scalar MSE loss.
    """
    return nn.functional.mse_loss(predicted, ground_truth)


def coverage_loss(
    soft_weights: torch.Tensor,
    spatial_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Coverage loss: encourages selected tokens to cover the task-relevant region.
    
    Computes a soft IoU-like measure between the weighted token distribution
    and the ground-truth spatial region.
    
    Args:
        soft_weights: (batch, N) soft selection weights from Gumbel-TopK.
        spatial_positions: (batch, N, 2) 2D positions of each visual token.
        target_positions: (batch, 2) target object/end-effector position.
    
    Returns:
        Scalar coverage loss (negative = better coverage).
    """
    # Distance from each token to target
    dists = torch.norm(spatial_positions - target_positions.unsqueeze(1), dim=-1)  # (batch, N)
    
    # Gaussian kernel: tokens closer to target should get higher weight
    sigma = 0.1  # spatial bandwidth
    relevance = torch.exp(-dists ** 2 / (2 * sigma ** 2))  # (batch, N)
    
    # Coverage: weighted sum of relevance
    coverage = (soft_weights * relevance).sum(dim=-1)  # (batch,)
    
    # Negative because we want to maximize coverage
    return -coverage.mean()


def diversity_loss(
    selected_indices: torch.Tensor,
    spatial_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Diversity loss: encourages spatial spread of selected tokens.
    
    Penalizes selecting only spatially adjacent tokens.
    
    Args:
        selected_indices: (batch, k) indices of selected tokens.
        spatial_positions: (batch, N, 2) 2D positions of each visual token.
    
    Returns:
        Scalar diversity loss (negative = more diverse).
    """
    batch_size, k = selected_indices.shape
    
    # Gather positions of selected tokens
    idx_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, 2)
    selected_pos = torch.gather(spatial_positions, 1, idx_expanded)  # (batch, k, 2)
    
    # Pairwise distances between selected tokens
    # (batch, k, 1, 2) - (batch, 1, k, 2) -> (batch, k, k)
    diffs = selected_pos.unsqueeze(2) - selected_pos.unsqueeze(1)
    pairwise_dists = torch.norm(diffs, dim=-1)
    
    # Mean pairwise distance (higher = more diverse)
    # Mask out diagonal (self-distances = 0)
    mask = 1.0 - torch.eye(k, device=pairwise_dists.device).unsqueeze(0)
    mean_dist = (pairwise_dists * mask).sum(dim=(1, 2)) / (k * (k - 1) + 1e-8)
    
    # Negative because we want to maximize diversity
    return -mean_dist.mean()


def apex_loss(
    predicted_action: torch.Tensor,
    ground_truth_action: torch.Tensor,
    soft_weights: torch.Tensor,
    selected_indices: torch.Tensor,
    spatial_positions: torch.Tensor | None = None,
    target_positions: torch.Tensor | None = None,
    lambda_coverage: float = 0.1,
    lambda_diversity: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined APEX loss.
    
    Args:
        predicted_action: (batch, 6) VLA action prediction.
        ground_truth_action: (batch, 6) ground-truth action.
        soft_weights: (batch, N) soft selection weights.
        selected_indices: (batch, k) selected token indices.
        spatial_positions: (batch, N, 2) optional spatial positions.
        target_positions: (batch, 2) optional target positions.
        lambda_coverage: Weight for coverage loss.
        lambda_diversity: Weight for diversity loss.
    
    Returns:
        total_loss: Scalar total loss.
        components: Dict of individual loss components for logging.
    """
    l_action = action_loss(predicted_action, ground_truth_action)
    components = {"action": l_action}
    
    total = l_action
    
    if spatial_positions is not None and target_positions is not None:
        l_coverage = coverage_loss(soft_weights, spatial_positions, target_positions)
        total = total + lambda_coverage * l_coverage
        components["coverage"] = l_coverage
    
    l_diversity = diversity_loss(selected_indices, spatial_positions)
    total = total + lambda_diversity * l_diversity
    components["diversity"] = l_diversity
    
    return total, components
