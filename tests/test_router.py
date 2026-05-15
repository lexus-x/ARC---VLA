"""Unit tests for APEX router."""

import torch
import pytest
from apex.router import APEXRouter
from apex.gumbel_topk import gumbel_topk, select_tokens
from apex.losses import action_loss, diversity_loss
from apex.utils import compute_spatial_positions


class TestAPEXRouter:
    def test_output_shape(self):
        router = APEXRouter(input_dim=256, hidden_dim=32)
        B, N, M, d = 2, 196, 20, 256
        visual = torch.randn(B, N, d)
        lang = torch.randn(B, M, d)
        weights = router(visual, lang)
        assert weights.shape == (B, N)
    
    def test_output_range(self):
        router = APEXRouter(input_dim=128, hidden_dim=16)
        visual = torch.randn(4, 100, 128)
        lang = torch.randn(4, 10, 128)
        weights = router(visual, lang)
        assert (weights >= 0).all() and (weights <= 1).all()
    
    def test_param_count(self):
        router = APEXRouter(input_dim=512, hidden_dim=64)
        params = sum(p.numel() for p in router.parameters())
        # 2*512*64 + 64 + 64 + 1 = 65,601
        assert params < 70_000
        assert params > 30_000


class TestGumbelTopK:
    def test_selection_count(self):
        weights = torch.rand(2, 196)
        indices, soft = gumbel_topk(weights, k=50, training=False)
        assert indices.shape == (2, 50)
    
    def test_unique_indices(self):
        weights = torch.rand(2, 196)
        indices, _ = gumbel_topk(weights, k=50, training=False)
        for b in range(2):
            assert len(set(indices[b].tolist())) == 50
    
    def test_token_gathering(self):
        tokens = torch.randn(2, 196, 512)
        weights = torch.rand(2, 196)
        indices, _ = gumbel_topk(weights, k=50, training=False)
        selected = select_tokens(tokens, indices)
        assert selected.shape == (2, 50, 512)


class TestLosses:
    def test_action_loss_zero(self):
        pred = torch.randn(4, 6)
        assert action_loss(pred, pred).item() < 1e-8
    
    def test_diversity_loss_shape(self):
        indices = torch.randint(0, 196, (2, 50))
        positions = compute_spatial_positions(196)
        positions = positions.unsqueeze(0).expand(2, -1, -1)
        loss = diversity_loss(indices, positions)
        assert loss.shape == ()
