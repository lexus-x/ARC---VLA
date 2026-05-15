"""
Basic usage example for APEX.

Shows how to wrap a VLA with APEX and run inference.
"""

import torch
from apex import APEXRouter, APEXWrappedVLA, wrap_vla


# --- Mock VLA for demonstration ---
class MockVLA(torch.nn.Module):
    """Mock VLA that takes visual tokens and produces 6-DoF actions."""
    
    def __init__(self, d_model: int = 512, n_layers: int = 4):
        super().__init__()
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=n_layers,
        )
        self.action_head = torch.nn.Linear(d_model, 6)
    
    def forward(self, visual_tokens, language_tokens, proprioception=None):
        # Concatenate all tokens
        tokens = torch.cat([language_tokens, visual_tokens], dim=1)
        # Transformer forward
        hidden = self.transformer(tokens)
        # Action from last token
        action = self.action_head(hidden[:, -1, :])
        return action


def main():
    # Hyperparameters
    batch_size = 4
    N = 196  # ViT tokens (14x14 patches for 224x224 image)
    M = 20   # Language tokens
    d = 512  # Token dimension
    
    # Create mock data
    visual_tokens = torch.randn(batch_size, N, d)
    language_tokens = torch.randn(batch_size, M, d)
    
    # Load frozen VLA
    vla = MockVLA(d_model=d)
    print(f"VLA parameters: {sum(p.numel() for p in vla.parameters()):,}")
    
    # Wrap with APEX (keep 25% of visual tokens)
    apex_vla = wrap_vla(vla, input_dim=d, target_ratio=0.25)
    
    apex_params = sum(p.numel() for p in apex_vla.router.parameters())
    print(f"APEX router parameters: {apex_params:,}")
    print(f"APEX overhead: {apex_params / sum(p.numel() for p in vla.parameters()) * 100:.3f}%")
    
    # Training mode
    apex_vla.train()
    action_train, info_train = apex_vla(visual_tokens, language_tokens, tau=1.0)
    print(f"\nTraining mode:")
    print(f"  Input tokens: {info_train['N']}")
    print(f"  Selected tokens: {info_train['k']}")
    print(f"  Compression ratio: {info_train['k'] / info_train['N']:.1%}")
    print(f"  Action shape: {action_train.shape}")
    
    # Inference mode
    apex_vla.eval()
    action_infer, info_infer = apex_vla(visual_tokens, language_tokens, tau=0.1)
    print(f"\nInference mode:")
    print(f"  Selected tokens: {info_infer['k']}")
    print(f"  Action shape: {action_infer.shape}")
    
    # Compare with full VLA
    with torch.no_grad():
        action_full = vla(visual_tokens, language_tokens)
    print(f"\nFull VLA action:  {action_full[0].tolist()}")
    print(f"APEX VLA action:  {action_infer[0].tolist()}")


if __name__ == "__main__":
    main()
