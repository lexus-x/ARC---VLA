"""
APEX VLA Wrapper: Wraps any frozen VLA with APEX token routing.

The wrapper:
1. Extracts visual and language tokens from the VLA's internal representations
2. Runs the APEX router to predict importance weights
3. Selects top-K tokens via Gumbel-TopK
4. Feeds only selected tokens to the frozen VLA
"""

import torch
import torch.nn as nn
from typing import Any, Protocol

from apex.router import APEXRouter
from apex.gumbel_topk import gumbel_topk, select_tokens


class FrozenVLA(Protocol):
    """Protocol for a frozen VLA model."""
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        proprioception: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict action from tokens."""
        ...


class APEXWrappedVLA(nn.Module):
    """
    Wraps a frozen VLA with APEX token routing.
    
    Args:
        vla: The frozen VLA model.
        input_dim: Token dimension (d).
        target_ratio: Fraction of visual tokens to keep (0.25 = keep 25%).
        hidden_dim: Router MLP hidden dimension.
        dropout: Router dropout probability.
    """
    
    def __init__(
        self,
        vla: nn.Module,
        input_dim: int = 512,
        target_ratio: float = 0.25,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vla = vla
        self.router = APEXRouter(input_dim, hidden_dim, dropout)
        self.target_ratio = target_ratio
        
        # Freeze the VLA
        for param in self.vla.parameters():
            param.requires_grad = False
    
    @property
    def target_k(self) -> int:
        """Number of tokens to select (computed dynamically)."""
        # This will be set based on actual N during first forward pass
        return getattr(self, "_target_k", None)
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        proprioception: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Forward pass with APEX routing.
        
        Args:
            visual_tokens: (batch, N, d) visual tokens from ViT.
            language_tokens: (batch, M, d) language tokens.
            proprioception: (batch, K) optional proprioceptive state.
            tau: Gumbel-Softmax temperature.
        
        Returns:
            action: (batch, 6) predicted action.
            info: Dict with routing info (indices, weights, etc.).
        """
        batch_size, N, d = visual_tokens.shape
        k = max(1, int(N * self.target_ratio))
        
        # Step 1: Compute importance weights
        weights = self.router(visual_tokens, language_tokens)  # (batch, N)
        
        # Step 2: Differentiable top-K selection
        selected_indices, soft_weights = gumbel_topk(
            weights, k, tau=tau, training=self.training
        )
        
        # Step 3: Select tokens
        selected_tokens = select_tokens(visual_tokens, selected_indices)  # (batch, k, d)
        
        # Step 4: Forward through frozen VLA with selected tokens
        action = self.vla(selected_tokens, language_tokens, proprioception)
        
        info = {
            "selected_indices": selected_indices,
            "weights": weights,
            "soft_weights": soft_weights,
            "k": k,
            "N": N,
        }
        
        return action, info
    
    def predict(
        self,
        visual_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        proprioception: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inference-mode prediction (deterministic selection).
        
        Returns:
            action: (batch, 6) predicted action.
        """
        self.eval()
        with torch.no_grad():
            action, _ = self.forward(
                visual_tokens, language_tokens, proprioception, tau=0.1
            )
        return action


def wrap_vla(
    vla: nn.Module,
    input_dim: int = 512,
    target_ratio: float = 0.25,
    hidden_dim: int = 64,
    dropout: float = 0.1,
) -> APEXWrappedVLA:
    """
    Convenience function to wrap a VLA with APEX.
    
    Args:
        vla: The frozen VLA model.
        input_dim: Token dimension.
        target_ratio: Fraction of visual tokens to keep.
        hidden_dim: Router hidden dimension.
        dropout: Router dropout.
    
    Returns:
        APEXWrappedVLA instance.
    """
    return APEXWrappedVLA(vla, input_dim, target_ratio, hidden_dim, dropout)
