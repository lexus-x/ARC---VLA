"""
APEX Router: Task-conditioned visual token importance predictor.

A lightweight MLP that takes each visual token concatenated with a
task (language) vector and predicts an importance weight in (0, 1).
"""

import torch
import torch.nn as nn


class APEXRouter(nn.Module):
    """
    Task-conditioned visual token router.
    
    For each visual token v_n, computes:
        w_n = sigmoid(W2 * ReLU(W1 * [v_n ; l_pool] + b1) + b2)
    
    where l_pool is the mean-pooled language token representation.
    
    Args:
        input_dim: Dimension of visual and language tokens (d).
        hidden_dim: Hidden layer dimension (default: 64).
        dropout: Dropout probability (default: 0.1).
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input is [v_n ; l_pool], so 2 * input_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token importance weights.
        
        Args:
            visual_tokens: (batch, N, d) visual token sequence from ViT.
            language_tokens: (batch, M, d) language token sequence.
        
        Returns:
            weights: (batch, N) importance weights in (0, 1).
        """
        # Step 1: Task vector - mean-pool language tokens
        l_pool = language_tokens.mean(dim=1)  # (batch, d)
        
        # Step 2: Expand l_pool to match visual token count
        N = visual_tokens.shape[1]
        l_pool_expanded = l_pool.unsqueeze(1).expand(-1, N, -1)  # (batch, N, d)
        
        # Step 3: Concatenate and predict importance
        x = torch.cat([visual_tokens, l_pool_expanded], dim=-1)  # (batch, N, 2d)
        logits = self.mlp(x).squeeze(-1)  # (batch, N)
        weights = torch.sigmoid(logits)  # (batch, N)
        
        return weights
    
    def extra_repr(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"params={total_params:,}"
        )
