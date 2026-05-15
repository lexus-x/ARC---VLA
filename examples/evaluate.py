"""
Evaluation script for APEX.

Demonstrates how to evaluate APEX routing decisions.
"""

import torch
from apex import APEXRouter
from apex.gumbel_topk import gumbel_topk
from apex.utils import compute_spatial_positions


def evaluate_routing():
    """Analyze APEX routing behavior."""
    
    # Config
    d = 256
    N = 196
    B = 4
    k = 50
    
    # Create router
    router = APEXRouter(input_dim=d, hidden_dim=64)
    
    # Mock tokens
    visual = torch.randn(B, N, d)
    lang = torch.randn(B, 20, d)
    
    # Compute importance weights
    router.eval()
    with torch.no_grad():
        weights = router(visual, lang)
    
    print("=== APEX Routing Analysis ===\n")
    print(f"Input tokens: {N}")
    print(f"Selected tokens: {k}")
    print(f"Compression ratio: {k/N:.1%}")
    print()
    
    # Analyze weight distribution
    print("Weight statistics:")
    print(f"  Mean: {weights.mean():.4f}")
    print(f"  Std:  {weights.std():.4f}")
    print(f"  Min:  {weights.min():.4f}")
    print(f"  Max:  {weights.max():.4f}")
    print()
    
    # Top-K selection
    indices, soft_weights = gumbel_topk(weights, k, training=False)
    
    # Spatial analysis
    positions = compute_spatial_positions(N)
    selected_positions = positions[indices[0]]
    
    print("Selected token positions (batch 0):")
    print(f"  X range: [{selected_positions[:, 0].min():.2f}, {selected_positions[:, 0].max():.2f}]")
    print(f"  Y range: [{selected_positions[:, 1].min():.2f}, {selected_positions[:, 1].max():.2f}]")
    print(f"  Centroid: ({selected_positions[:, 0].mean():.2f}, {selected_positions[:, 1].mean():.2f})")
    print()
    
    # Spatial spread (diversity)
    pairwise_dists = torch.cdist(selected_positions.unsqueeze(0), selected_positions.unsqueeze(0))
    mean_dist = pairwise_dists[0].mean().item()
    print(f"Mean pairwise distance: {mean_dist:.3f}")
    print(f"(Higher = more spatially diverse)")
    
    # Per-sample comparison
    print("\n=== Per-sample analysis ===")
    for b in range(min(B, 4)):
        w = weights[b]
        top5 = torch.topk(w, 5).indices.tolist()
        bottom5 = torch.topk(w, 5, largest=False).indices.tolist()
        print(f"\nBatch {b}:")
        print(f"  Top-5 token indices: {top5}")
        print(f"  Bottom-5 token indices: {bottom5}")
        print(f"  Weight range: [{w.min():.4f}, {w.max():.4f}]")


if __name__ == "__main__":
    evaluate_routing()
