"""Unit tests for AEGIS-VLA module."""

import torch

from aegis.energy_head import InteractionEnergyNetwork, IENConfig
from aegis.criticality import CriticalityEstimator, adaptive_num_steps
from aegis.geometric_tokens import GeometricInteractionTokens
from aegis.aegis_module import AEGISActionHead, AEGISConfig
from aegis.losses import (
    energy_contrastive_loss,
    amortised_loss,
    criticality_loss,
    consistency_loss,
    aegis_loss,
)
from aegis.utils import make_negative_samples, EMAModel


# ── Fixtures ──────────────────────────────────────────────────────────
B, N, M, D, A, P = 4, 49, 10, 128, 7, 7  # small dims for fast tests


def _ctx() -> torch.Tensor:
    return torch.randn(B, N + M, D)


def _vis() -> torch.Tensor:
    return torch.randn(B, N, D)


def _lang() -> torch.Tensor:
    return torch.randn(B, M, D)


def _proprio() -> torch.Tensor:
    return torch.randn(B, P)


def _action() -> torch.Tensor:
    return torch.randn(B, A)


def _small_cfg() -> AEGISConfig:
    return AEGISConfig(
        token_dim=D,
        action_dim=A,
        proprio_dim=P,
        ien=IENConfig(
            token_dim=D,
            action_dim=A,
            num_heads=2,
            num_energy_layers=1,
            mlp_hidden=64,
            dropout=0.0,
        ),
        num_geo_queries=2,
        geo_num_heads=2,
        crit_hidden=32,
        min_refine_steps=1,
        max_refine_steps=3,
        amort_hidden=64,
        dropout=0.0,
        num_negative_samples=8,
    )


# ── InteractionEnergyNetwork ─────────────────────────────────────────
class TestIEN:
    def test_energy_shape(self):
        cfg = IENConfig(token_dim=D, action_dim=A, num_heads=2,
                        num_energy_layers=1, mlp_hidden=64)
        ien = InteractionEnergyNetwork(cfg)
        actions = torch.randn(B, 16, A)
        ctx = _ctx()
        energies = ien(actions, ctx)
        assert energies.shape == (B, 16)

    def test_single_energy(self):
        cfg = IENConfig(token_dim=D, action_dim=A, num_heads=2,
                        num_energy_layers=1, mlp_hidden=64)
        ien = InteractionEnergyNetwork(cfg)
        e = ien.single_energy(_action(), _ctx())
        assert e.shape == (B,)

    def test_energy_gradient(self):
        cfg = IENConfig(token_dim=D, action_dim=A, num_heads=2,
                        num_energy_layers=1, mlp_hidden=64)
        ien = InteractionEnergyNetwork(cfg)
        grad, energy = ien.energy_gradient(_action(), _ctx())
        assert grad.shape == (B, A)
        assert energy.shape == (B,)

    def test_energy_differentiable(self):
        cfg = IENConfig(token_dim=D, action_dim=A, num_heads=2,
                        num_energy_layers=1, mlp_hidden=64)
        ien = InteractionEnergyNetwork(cfg)
        actions = _action().requires_grad_(True)
        e = ien.single_energy(actions, _ctx())
        e.sum().backward()
        assert actions.grad is not None


# ── CriticalityEstimator ─────────────────────────────────────────────
class TestCriticality:
    def test_output_shape(self):
        ce = CriticalityEstimator(token_dim=D, proprio_dim=P, hidden_dim=32)
        c = ce(_vis(), _proprio())
        assert c.shape == (B,)

    def test_output_range(self):
        ce = CriticalityEstimator(token_dim=D, proprio_dim=P, hidden_dim=32)
        c = ce(_vis(), _proprio())
        assert (c >= 0).all() and (c <= 1).all()

    def test_target_criticality(self):
        grad_norm = torch.rand(B)
        target = CriticalityEstimator.compute_target_criticality(grad_norm)
        assert (target >= 0).all() and (target <= 1).all()

    def test_adaptive_steps(self):
        crit = torch.tensor([0.0, 0.5, 1.0])
        steps = adaptive_num_steps(crit, min_steps=1, max_steps=8)
        assert steps[0].item() == 1
        assert steps[2].item() == 8
        assert 1 <= steps[1].item() <= 8


# ── GeometricInteractionTokens ───────────────────────────────────────
class TestGIT:
    def test_output_shape(self):
        git = GeometricInteractionTokens(
            token_dim=D, proprio_dim=P, num_queries=4, num_heads=2,
        )
        geo = git(_vis(), _proprio())
        assert geo.shape == (B, 4, D)

    def test_gradient_flow(self):
        git = GeometricInteractionTokens(
            token_dim=D, proprio_dim=P, num_queries=2, num_heads=2,
        )
        vis = _vis().requires_grad_(True)
        geo = git(vis, _proprio())
        geo.sum().backward()
        assert vis.grad is not None


# ── AEGISActionHead ──────────────────────────────────────────────────
class TestAEGIS:
    def test_forward_training(self):
        cfg = _small_cfg()
        head = AEGISActionHead(cfg)
        out = head(_vis(), _lang(), _proprio(), gt_action=_action())
        assert out["action_init"].shape == (B, A)
        assert out["action_refined"].shape == (B, A)
        assert out["criticality"].shape == (B,)
        assert "energy_gt" in out
        assert "energy_neg" in out
        assert out["energy_neg"].shape == (B, cfg.num_negative_samples)

    def test_predict(self):
        cfg = _small_cfg()
        head = AEGISActionHead(cfg)
        action = head.predict(_vis(), _lang(), _proprio())
        assert action.shape == (B, A)

    def test_param_summary(self):
        cfg = _small_cfg()
        head = AEGISActionHead(cfg)
        summary = head.param_summary()
        assert "total" in summary
        assert summary["total"] > 0
        assert summary["total"] == sum(
            v for k, v in summary.items() if k != "total"
        )

    def test_forward_no_gt(self):
        cfg = _small_cfg()
        head = AEGISActionHead(cfg)
        out = head(_vis(), _lang(), _proprio(), gt_action=None)
        assert "action_init" in out
        assert "energy_gt" not in out


# ── Losses ────────────────────────────────────────────────────────────
class TestLosses:
    def test_energy_contrastive(self):
        energy_gt = torch.randn(B)
        energy_neg = torch.randn(B, 16)
        loss = energy_contrastive_loss(energy_gt, energy_neg)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_amortised_loss_zero(self):
        a = _action()
        assert amortised_loss(a, a).item() < 1e-7

    def test_criticality_loss(self):
        pred = torch.sigmoid(torch.randn(B))
        grad_norm = torch.rand(B)
        loss = criticality_loss(pred, grad_norm)
        assert loss.shape == ()

    def test_consistency_loss_none(self):
        loss = consistency_loss(_action(), None)
        assert loss.item() == 0.0

    def test_consistency_loss_nonzero(self):
        a1 = _action()
        a2 = _action()
        loss = consistency_loss(a1, a2)
        assert loss.item() > 0

    def test_aegis_loss_full(self):
        cfg = _small_cfg()
        head = AEGISActionHead(cfg)
        gt = _action()
        out = head(_vis(), _lang(), _proprio(), gt_action=gt)
        total, components = aegis_loss(out, gt)
        assert total.shape == ()
        assert set(components.keys()) == {
            "energy", "amortised", "criticality", "consistency", "total",
        }


# ── Utils ─────────────────────────────────────────────────────────────
class TestUtils:
    def test_negative_samples(self):
        gt = _action()
        negs = make_negative_samples(gt, num_samples=32)
        assert negs.shape == (B, 32, A)

    def test_ema(self):
        cfg = _small_cfg()
        head = AEGISActionHead(cfg)
        ema = EMAModel(head, decay=0.99)
        # Perturb params
        for p in head.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
        ema.update(head)
        # Shadow should differ from current params
        for name, param in head.named_parameters():
            if name in ema.shadow:
                assert not torch.allclose(param.data, ema.shadow[name])
