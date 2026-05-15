"""
Training script for APEX.

Demonstrates how to train the APEX router on a VLA's training data.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from apex import APEXRouter, APEXWrappedVLA, wrap_vla
from apex.losses import apex_loss
from apex.utils import cosine_anneal, compute_spatial_positions, count_parameters, format_params


# --- Mock VLA ---
class MockVLA(torch.nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d, nhead=8, batch_first=True),
            num_layers=4,
        )
        self.action_head = torch.nn.Linear(d, 6)
    
    def forward(self, visual_tokens, language_tokens, proprioception=None):
        tokens = torch.cat([language_tokens, visual_tokens], dim=1)
        hidden = self.transformer(tokens)
        return self.action_head(hidden[:, -1, :])


def create_mock_dataset(n_samples=500, N=196, M=20, d=256):
    """Create synthetic training data."""
    visual = torch.randn(n_samples, N, d)
    lang = torch.randn(n_samples, M, d)
    actions = torch.randn(n_samples, 6)
    return TensorDataset(visual, lang, actions)


def train_apex():
    # Config
    d = 256
    N = 196
    target_ratio = 0.25
    epochs = 5
    lr = 3e-4
    batch_size = 16
    
    # Create mock data
    dataset = create_mock_dataset(n_samples=500, N=N, M=20, d=d)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create VLA and wrap with APEX
    vla = MockVLA(d=d)
    apex_vla = wrap_vla(vla, input_dim=d, target_ratio=target_ratio)
    
    print(f"VLA parameters: {format_params(count_parameters(vla))}")
    print(f"APEX router parameters: {format_params(count_parameters(apex_vla.router))}")
    print(f"Target tokens: {int(N * target_ratio)} / {N}")
    print()
    
    # Only train the router
    optimizer = torch.optim.AdamW(apex_vla.router.parameters(), lr=lr, weight_decay=0.01)
    
    spatial_positions = compute_spatial_positions(N).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Training loop
    apex_vla.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (visual, lang, gt_action) in enumerate(dataloader):
            B = visual.shape[0]
            
            # Update spatial positions for current batch size
            sp = compute_spatial_positions(N).unsqueeze(0).expand(B, -1, -1)
            
            # Anneal Gumbel temperature
            step = epoch * len(dataloader) + batch_idx
            total_steps = epochs * len(dataloader)
            tau = cosine_anneal(step, total_steps, start=1.0, end=0.1)
            
            # Forward
            pred_action, info = apex_vla(visual, lang, tau=tau)
            
            # Loss
            target_pos = torch.rand(B, 2)  # Mock target positions
            loss, components = apex_loss(
                pred_action, gt_action,
                info["soft_weights"], info["selected_indices"],
                spatial_positions=sp[:B],
                target_positions=target_pos,
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | tau: {tau:.3f}")
    
    print("\nTraining complete!")
    print(f"Final router: {apex_vla.router}")


if __name__ == "__main__":
    train_apex()
