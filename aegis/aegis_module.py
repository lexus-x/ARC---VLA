"""
AEGIS Action Head — the unified module.

Replaces a standard VLA action decoder (regression MLP, discrete
tokeniser, or diffusion head) with energy-based action generation
that features criticality-aware adaptive computation and geometric
interaction tokens.

Inference pipeline
------------------
1. Extract Geometric Interaction Tokens from visual tokens + proprio.
2. Build context = [visual_tokens ; language_tokens ; geo_tokens].
3. Estimate criticality c from visual tokens + proprio.
4. Produce an initial action via the amortised predictor (single pass).
5. Refine the action via Langevin dynamics on the energy landscape,
   running  N = f(c)  refinement steps.
6. Return the refined action.

Training pipeline
-----------------
The module is trained end-to-end with four loss terms (see losses.py):
* L_energy    — InfoNCE contrastive energy loss
* L_amortised — MSE on the warm-start predictor
* L_crit      — self-supervised criticality prediction
* L_consist   — temporal action consistency
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from aegis.energy_head import InteractionEnergyNetwork, IENConfig
from aegis.criticality import CriticalityEstimator, adaptive_num_steps
from aegis.geometric_tokens import GeometricInteractionTokens


@dataclass
class AEGISConfig:
    """Full configuration for the AEGIS action head."""

    # Dimensions
    token_dim: int = 512
    action_dim: int = 7
    proprio_dim: int = 7

    # Energy network
    ien: IENConfig = field(default_factory=lambda: IENConfig())

    # Geometric tokens
    num_geo_queries: int = 4
    geo_num_heads: int = 4

    # Criticality
    crit_hidden: int = 128
    min_refine_steps: int = 1
    max_refine_steps: int = 8

    # Langevin dynamics
    langevin_lr: float = 0.01
    langevin_noise_scale: float = 0.005
    langevin_grad_clip: float = 1.0

    # Amortised predictor
    amort_hidden: int = 256

    # Training
    dropout: float = 0.1
    num_negative_samples: int = 64

    def __post_init__(self) -> None:
        self.ien = IENConfig(
            token_dim=self.token_dim,
            action_dim=self.action_dim,
            num_heads=self.ien.num_heads,
            num_energy_layers=self.ien.num_energy_layers,
            mlp_hidden=self.ien.mlp_hidden,
            dropout=self.dropout,
        )


class AmortisedPredictor(nn.Module):
    """
    Single-pass action predictor that warm-starts the Langevin sampler.

    For low-criticality timesteps (free-space motion) this output is
    used directly without refinement.
    """

    def __init__(self, token_dim: int, action_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, C, D)  fused visual-language-geometric context.

        Returns:
            action: (B, A)  initial action prediction.
        """
        B = context.shape[0]
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, self.norm(context), self.norm(context))
        return self.mlp(pooled.squeeze(1))


class AEGISActionHead(nn.Module):
    """
    Full AEGIS action head.

    Combines:
    * GeometricInteractionTokens  (spatial grounding)
    * CriticalityEstimator        (adaptive compute)
    * InteractionEnergyNetwork    (energy landscape)
    * AmortisedPredictor          (warm-start)
    * Langevin dynamics            (iterative refinement)

    Args:
        cfg: AEGISConfig.
    """

    def __init__(self, cfg: AEGISConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or AEGISConfig()
        self.cfg = cfg

        # Sub-modules
        self.geo_tokens = GeometricInteractionTokens(
            token_dim=cfg.token_dim,
            proprio_dim=cfg.proprio_dim,
            num_queries=cfg.num_geo_queries,
            num_heads=cfg.geo_num_heads,
            dropout=cfg.dropout,
        )
        self.criticality = CriticalityEstimator(
            token_dim=cfg.token_dim,
            proprio_dim=cfg.proprio_dim,
            hidden_dim=cfg.crit_hidden,
            dropout=cfg.dropout,
        )
        self.energy_net = InteractionEnergyNetwork(cfg.ien)
        self.amortised = AmortisedPredictor(
            token_dim=cfg.token_dim,
            action_dim=cfg.action_dim,
            hidden=cfg.amort_hidden,
            dropout=cfg.dropout,
        )

    # ------------------------------------------------------------------
    #  Context construction
    # ------------------------------------------------------------------
    def _build_context(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate visual + language + geometric tokens."""
        geo = self.geo_tokens(visual_tokens, proprioception)  # (B, G, D)
        context = torch.cat([visual_tokens, language_tokens, geo], dim=1)
        return context

    # ------------------------------------------------------------------
    #  Langevin refinement
    # ------------------------------------------------------------------
    def _langevin_step(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single Langevin MCMC step:
            a <- a - lr * dE/da + sqrt(2*lr) * noise
        """
        with torch.enable_grad():
            grad, energy = self.energy_net.energy_gradient(action, context)
        grad = grad.detach().clamp(
            -self.cfg.langevin_grad_clip, self.cfg.langevin_grad_clip,
        )

        lr = self.cfg.langevin_lr
        noise_scale = self.cfg.langevin_noise_scale
        noise = torch.randn_like(action) * noise_scale

        refined = action.detach() - lr * grad + noise
        return refined, energy.detach()

    def _refine(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Run multiple Langevin steps (uniform count for whole batch)."""
        for _ in range(num_steps):
            action, _ = self._langevin_step(action, context)
        return action

    def _refine_adaptive(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
        step_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-sample adaptive refinement.

        Runs up to max(step_counts) iterations; at each iteration only
        the samples that still have remaining steps are updated.
        """
        max_steps = step_counts.max().item()
        for s in range(max_steps):
            mask = (step_counts > s).float().unsqueeze(-1)  # (B, 1)
            refined, _ = self._langevin_step(action, context)
            action = mask * refined + (1.0 - mask) * action
        return action

    # ------------------------------------------------------------------
    #  Forward (training)
    # ------------------------------------------------------------------
    def forward(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        proprioception: torch.Tensor,
        gt_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Returns a dict of tensors needed by aegis_loss():
            - action_init:     amortised action prediction
            - action_refined:  Langevin-refined action
            - criticality:     predicted criticality score
            - energy_gt:       energy at ground-truth action
            - energy_neg:      energies at negative samples
            - energy_grad_norm: || dE/da || at GT action (for crit target)
            - context:         fused context (for external use)
        """
        B = visual_tokens.shape[0]
        ctx = self._build_context(visual_tokens, language_tokens, proprioception)

        # Amortised warm-start
        action_init = self.amortised(ctx)  # (B, A)

        # Criticality
        crit = self.criticality(visual_tokens, proprioception)  # (B,)
        step_counts = adaptive_num_steps(
            crit,
            self.cfg.min_refine_steps,
            self.cfg.max_refine_steps,
        )

        # Langevin refinement from amortised init
        action_refined = self._refine_adaptive(
            action_init.detach(), ctx, step_counts,
        )

        result: dict[str, torch.Tensor] = {
            "action_init": action_init,
            "action_refined": action_refined,
            "criticality": crit,
            "context": ctx,
        }

        # Energy scores for contrastive loss (requires GT action)
        if gt_action is not None:
            # Energy at GT action
            grad_gt, energy_gt = self.energy_net.energy_gradient(gt_action, ctx)
            grad_norm = grad_gt.norm(dim=-1)  # (B,)

            # Negative samples: perturbed GT actions
            noise = torch.randn(
                B, self.cfg.num_negative_samples, self.cfg.action_dim,
                device=gt_action.device,
            ) * 0.1
            negatives = gt_action.unsqueeze(1) + noise  # (B, K, A)
            energy_neg = self.energy_net(negatives, ctx)  # (B, K)

            result["energy_gt"] = energy_gt
            result["energy_neg"] = energy_neg
            result["energy_grad_norm"] = grad_norm

        return result

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference: amortised init + adaptive Langevin refinement.

        Returns:
            action: (B, A)  refined action.
        """
        self.eval()
        ctx = self._build_context(visual_tokens, language_tokens, proprioception)

        action = self.amortised(ctx)
        crit = self.criticality(visual_tokens, proprioception)
        steps = adaptive_num_steps(
            crit,
            self.cfg.min_refine_steps,
            self.cfg.max_refine_steps,
        )
        action = self._refine_adaptive(action, ctx, steps)
        return action

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def param_summary(self) -> dict[str, int]:
        """Count parameters per sub-module."""
        def _count(mod: nn.Module) -> int:
            return sum(p.numel() for p in mod.parameters())

        return {
            "geo_tokens": _count(self.geo_tokens),
            "criticality": _count(self.criticality),
            "energy_net": _count(self.energy_net),
            "amortised": _count(self.amortised),
            "total": _count(self),
        }
