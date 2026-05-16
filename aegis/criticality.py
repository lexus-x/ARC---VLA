"""
Criticality-Aware Adaptive Computation (CAAC).

Estimates a per-timestep *criticality score*  c in [0, 1]  that
indicates how much computational effort the action-generation step
requires.

Criticality semantics
---------------------
* c ~ 0  --  free-space motion (reaching toward an object).  The energy
  landscape is smooth and a single amortised forward pass suffices.
* c ~ 1  --  contact-rich phase (grasping, inserting, rotating under
  contact).  The energy landscape is rugged and multi-modal; several
  Langevin refinement steps are needed.

Self-supervised training signal
-------------------------------
Criticality is *not* manually labelled.  Instead we derive it from the
geometry of the energy landscape itself:

    c* = sigmoid( alpha * || dE/da ||_2 )

where  dE/da  is the gradient of the energy at the ground-truth action.
High gradient magnitude means the landscape is steep near the optimum,
i.e. small action perturbations cause large energy changes -- exactly
the situations that need careful refinement.

The estimator is trained to predict  c*  from the observation alone
(no access to the ground-truth action at test time).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CriticalityEstimator(nn.Module):
    """
    Predicts per-timestep criticality from visual features and
    proprioceptive state.

    Architecture: attention-pooled visual features + proprio -> MLP -> c.

    Args:
        token_dim:   Dimension of visual tokens.
        proprio_dim: Dimension of proprioceptive state vector.
        hidden_dim:  MLP hidden dimension.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        token_dim: int = 512,
        proprio_dim: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim

        # Learned attention-pooling query
        self.pool_query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(token_dim)

        # MLP: pooled_visual + proprio -> criticality
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
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
            criticality: (B,)  in [0, 1].
        """
        B = visual_tokens.shape[0]

        # Attention-pool visual tokens into a single vector
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.pool_attn(
            query,
            self.pool_norm(visual_tokens),
            self.pool_norm(visual_tokens),
        )  # (B, 1, D)
        pooled = pooled.squeeze(1)  # (B, D)

        # Concatenate with proprioception and predict criticality
        combined = torch.cat([pooled, proprioception], dim=-1)  # (B, D+P)
        logit = self.mlp(combined).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)

    @staticmethod
    def compute_target_criticality(
        energy_grad_norm: torch.Tensor,
        alpha: float = 5.0,
    ) -> torch.Tensor:
        """
        Self-supervised criticality target from energy gradient magnitude.

        Args:
            energy_grad_norm: (B,)  || dE/da ||_2  at the GT action.
            alpha: Scaling factor before sigmoid.

        Returns:
            target: (B,) in [0, 1].
        """
        return torch.sigmoid(alpha * energy_grad_norm)


def adaptive_num_steps(
    criticality: torch.Tensor,
    min_steps: int = 1,
    max_steps: int = 8,
) -> torch.Tensor:
    """
    Map criticality scores to integer refinement step counts.

    Uses a linear mapping:  n = min_steps + round(c * (max_steps - min_steps))

    Args:
        criticality: (B,)  values in [0, 1].
        min_steps:   Minimum refinement steps (for easy actions).
        max_steps:   Maximum refinement steps (for hard actions).

    Returns:
        steps: (B,)  integer step counts.
    """
    continuous = min_steps + criticality * (max_steps - min_steps)
    return continuous.round().long().clamp(min=min_steps, max=max_steps)
