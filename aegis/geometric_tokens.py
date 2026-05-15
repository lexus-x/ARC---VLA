"""
Geometric Interaction Tokens (GIT).

Extracts compact geometric descriptors that encode the spatial
relationship between the end-effector and task-relevant objects,
providing explicit geometric grounding to the action head.

Design motivation
-----------------
Standard VLAs receive only 2-D visual tokens from a ViT.  These tokens
encode appearance but lack explicit 3-D geometric structure.  GIT
addresses this by:

1. Learning a small set of *geometric queries* (G << N visual tokens)
   that cross-attend to the visual tokens conditioned on the robot's
   proprioceptive state (end-effector pose).
2. Outputting compact *geometric interaction tokens* that capture
   where the end-effector is relative to task-relevant objects.

These tokens are injected into the energy network's context alongside
the standard visual and language tokens, enriching the action head with
spatial information without requiring depth sensors or 3-D point clouds.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeometricInteractionTokens(nn.Module):
    """
    Produces G geometric interaction tokens by cross-attending learned
    queries to visual tokens, conditioned on proprioception.

    Args:
        token_dim:    Dimension of visual / output tokens.
        proprio_dim:  Dimension of proprioceptive state vector.
        num_queries:  Number of geometric tokens to produce (G).
        num_heads:    Multi-head attention heads.
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        token_dim: int = 512,
        proprio_dim: int = 7,
        num_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.token_dim = token_dim

        # Learned geometric queries
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, token_dim) * 0.02
        )

        # Proprioception-conditioned bias for the queries
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_queries * token_dim),
        )

        # Cross-attention: geometric queries attend to visual tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)

        # Feed-forward refinement
        self.ff = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens:  (B, N, D)
            proprioception: (B, P)

        Returns:
            geo_tokens: (B, G, D)  geometric interaction tokens.
        """
        B = visual_tokens.shape[0]

        # Condition queries on proprioception
        proprio_bias = self.proprio_proj(proprioception)  # (B, G*D)
        proprio_bias = proprio_bias.view(B, self.num_queries, self.token_dim)
        queries = self.queries.expand(B, -1, -1) + proprio_bias  # (B, G, D)

        # Cross-attend to visual tokens
        queries_normed = self.norm_q(queries)
        vis_normed = self.norm_kv(visual_tokens)
        attended, _ = self.cross_attn(
            queries_normed, vis_normed, vis_normed,
        )  # (B, G, D)

        # Residual + feed-forward
        geo_tokens = queries + attended
        geo_tokens = geo_tokens + self.ff(geo_tokens)

        return geo_tokens
