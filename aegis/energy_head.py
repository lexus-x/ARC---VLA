"""
Interaction Energy Network (IEN).

Learns an energy function  E(a | o, l) : R^A -> R  that maps candidate
actions to scalar energy values conditioned on visual observation and
language instruction.  Low energy = high-quality action.

Key design choices
------------------
* **Action-conditioned cross-attention**: candidate actions attend to
  fused visual-language context, enabling the energy landscape to be
  shaped by both task semantics and scene geometry.
* **Spectral normalisation** on all linear layers to ensure the energy
  is Lipschitz-continuous, which stabilises Langevin sampling at
  inference time.
* **Negative-sample contrastive training** (InfoNCE): ground-truth
  actions receive low energy; randomly perturbed negatives receive high
  energy.  This avoids mode collapse and learns a multi-modal landscape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class IENConfig:
    """Configuration for the Interaction Energy Network."""

    token_dim: int = 512
    action_dim: int = 7
    num_heads: int = 4
    num_energy_layers: int = 2
    mlp_hidden: int = 256
    dropout: float = 0.1


class SpectralLinear(nn.Module):
    """Linear layer with spectral normalisation."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.utils.parametrizations.spectral_norm(
            nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ActionContextCrossAttention(nn.Module):
    """Action queries attend to visual-language context."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = SpectralLinear(dim, dim)
        self.k_proj = SpectralLinear(dim, dim)
        self.v_proj = SpectralLinear(dim, dim)
        self.out_proj = SpectralLinear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:   (B, Q, D)  -- action queries
            context: (B, C, D)  -- visual-language context
        Returns:
            (B, Q, D)
        """
        query = self.norm_q(query)
        context = self.norm_kv(context)

        B, Q, D = query.shape
        H = self.num_heads

        q = self.q_proj(query).view(B, Q, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, -1, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, -1, H, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Q, D)
        return self.out_proj(out)


class EnergyBlock(nn.Module):
    """Single transformer block for the energy network."""

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = ActionContextCrossAttention(dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            SpectralLinear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            SpectralLinear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = query + self.cross_attn(query, context)
        query = query + self.ff(query)
        return query


class InteractionEnergyNetwork(nn.Module):
    """
    Scores candidate actions via a learned energy landscape.

    E(a | o, l) = MLP( CrossAttn( proj(a), [o; l] ) )

    Low energy indicates a high-quality action.

    Args:
        cfg: IENConfig with architecture hyperparameters.
    """

    def __init__(self, cfg: IENConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or IENConfig()
        self.cfg = cfg

        # Project raw action vector into token space
        self.action_proj = nn.Sequential(
            SpectralLinear(cfg.action_dim, cfg.token_dim),
            nn.GELU(),
            SpectralLinear(cfg.token_dim, cfg.token_dim),
        )

        # Stacked cross-attention blocks
        self.blocks = nn.ModuleList([
            EnergyBlock(cfg.token_dim, cfg.num_heads, cfg.mlp_hidden, cfg.dropout)
            for _ in range(cfg.num_energy_layers)
        ])

        # Scalar energy readout
        self.energy_head = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            SpectralLinear(cfg.token_dim, cfg.mlp_hidden),
            nn.GELU(),
            SpectralLinear(cfg.mlp_hidden, 1),
        )

    def forward(
        self,
        actions: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy for candidate actions.

        Args:
            actions: (B, K, A) batch of K candidate action vectors.
            context: (B, C, D) fused visual-language context tokens.

        Returns:
            energies: (B, K) scalar energy per candidate.
        """
        B, K, A = actions.shape

        # Project actions into token space -> (B, K, D)
        action_tokens = self.action_proj(actions)

        # Cross-attend to context
        for block in self.blocks:
            action_tokens = block(action_tokens, context)

        # Scalar readout per candidate
        energies = self.energy_head(action_tokens).squeeze(-1)  # (B, K)
        return energies

    def single_energy(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Energy for a single action per batch element.

        Args:
            action:  (B, A)
            context: (B, C, D)

        Returns:
            energy: (B,)
        """
        return self.forward(action.unsqueeze(1), context).squeeze(1)

    def energy_gradient(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute  dE/da  for Langevin dynamics.

        Args:
            action:  (B, A)  -- requires_grad will be set internally.
            context: (B, C, D)

        Returns:
            grad:   (B, A)  gradient of energy w.r.t. action.
            energy: (B,)    scalar energy values.
        """
        action = action.detach().requires_grad_(True)
        energy = self.single_energy(action, context)
        grad = torch.autograd.grad(
            energy.sum(), action, create_graph=self.training
        )[0]
        return grad, energy
