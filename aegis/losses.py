"""
AEGIS training losses.

Four components, jointly optimised:

1. L_energy       -- InfoNCE contrastive energy
2. L_amortised    -- MSE on the warm-start predictor
3. L_criticality  -- self-supervised criticality prediction
4. L_consistency  -- temporal action smoothness
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
#  1.  Contrastive energy loss  (InfoNCE)
# ------------------------------------------------------------------
def energy_contrastive_loss(
    energy_gt: torch.Tensor,
    energy_neg: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE: ground-truth action should have the *lowest* energy.

    L = -log( exp(-E_gt / tau) / (exp(-E_gt / tau) + sum exp(-E_neg / tau)) )

    Args:
        energy_gt:  (B,)    energy at ground-truth action.
        energy_neg: (B, K)  energies at K negative samples.
        temperature: softmax temperature.

    Returns:
        Scalar loss.
    """
    # Stack gt as first column: (B, 1+K)
    logits = torch.cat([energy_gt.unsqueeze(1), energy_neg], dim=1)
    logits = -logits / temperature           # negate: low energy = high logit
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


# ------------------------------------------------------------------
#  2.  Amortised predictor loss
# ------------------------------------------------------------------
def amortised_loss(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    """MSE between amortised action prediction and GT."""
    return F.mse_loss(predicted, ground_truth)


# ------------------------------------------------------------------
#  3.  Self-supervised criticality loss
# ------------------------------------------------------------------
def criticality_loss(
    predicted_crit: torch.Tensor,
    energy_grad_norm: torch.Tensor,
    alpha: float = 5.0,
) -> torch.Tensor:
    """
    BCE between predicted criticality and the self-supervised target
    derived from energy gradient magnitude.

    Args:
        predicted_crit:    (B,)  predicted c in [0,1].
        energy_grad_norm:  (B,)  || dE/da || at GT action.
        alpha: scaling before sigmoid for the target.

    Returns:
        Scalar loss.
    """
    target = torch.sigmoid(alpha * energy_grad_norm).detach()
    return F.binary_cross_entropy(predicted_crit, target)


# ------------------------------------------------------------------
#  4.  Temporal action consistency loss
# ------------------------------------------------------------------
def consistency_loss(
    action_t: torch.Tensor,
    action_t_minus1: torch.Tensor | None,
) -> torch.Tensor:
    """
    Penalises large jumps between consecutive predicted actions.

    L = || a_t - a_{t-1} ||^2

    Args:
        action_t:        (B, A)  current action.
        action_t_minus1: (B, A)  previous action (None if first step).

    Returns:
        Scalar loss (0 if no previous action).
    """
    if action_t_minus1 is None:
        return torch.tensor(0.0, device=action_t.device)
    return F.mse_loss(action_t, action_t_minus1)


# ------------------------------------------------------------------
#  Combined loss
# ------------------------------------------------------------------
def aegis_loss(
    output: dict[str, torch.Tensor],
    gt_action: torch.Tensor,
    prev_action: torch.Tensor | None = None,
    lambda_energy: float = 1.0,
    lambda_amort: float = 1.0,
    lambda_crit: float = 0.5,
    lambda_consist: float = 0.1,
    energy_temperature: float = 0.1,
    crit_alpha: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Full AEGIS loss.

    Args:
        output:      Dict returned by AEGISActionHead.forward().
        gt_action:   (B, A)  ground-truth action.
        prev_action: (B, A)  previous action (for consistency).
        lambda_*:    Loss weights.
        energy_temperature: InfoNCE temperature.
        crit_alpha:  Criticality target scaling.

    Returns:
        total: scalar total loss.
        components: dict of named scalar components for logging.
    """
    components: dict[str, torch.Tensor] = {}

    # Energy contrastive
    l_energy = energy_contrastive_loss(
        output["energy_gt"], output["energy_neg"], energy_temperature,
    )
    components["energy"] = l_energy

    # Amortised predictor
    l_amort = amortised_loss(output["action_init"], gt_action)
    components["amortised"] = l_amort

    # Criticality
    l_crit = criticality_loss(
        output["criticality"], output["energy_grad_norm"], crit_alpha,
    )
    components["criticality"] = l_crit

    # Consistency
    l_consist = consistency_loss(output["action_refined"], prev_action)
    components["consistency"] = l_consist

    total = (
        lambda_energy * l_energy
        + lambda_amort * l_amort
        + lambda_crit * l_crit
        + lambda_consist * l_consist
    )
    components["total"] = total

    return total, components
