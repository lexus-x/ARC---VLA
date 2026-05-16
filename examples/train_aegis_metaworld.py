"""
AEGIS-VLA training on Meta-World.

This script trains the AEGIS action head on expert demonstrations
collected from Meta-World environments.  It uses a lightweight ViT
encoder for visual tokens and a sentence-transformer for language
tokens, both frozen, with only the AEGIS module trained.

Usage
-----
# Collect demos first (or use pre-collected):
python examples/collect_metaworld_demos.py --tasks MT10 --episodes 50

# Train AEGIS:
python examples/train_aegis_metaworld.py \
    --demo_dir data/metaworld_demos \
    --tasks MT10 \
    --epochs 100 \
    --batch_size 64 \
    --lr 3e-4 \
    --save_dir checkpoints/aegis_mt10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from aegis.aegis_module import AEGISActionHead, AEGISConfig
from aegis.energy_head import IENConfig
from aegis.losses import aegis_loss
from aegis.utils import format_params, EMAModel


# ── Lightweight frozen encoders ───────────────────────────────────────

class SimpleVisualEncoder(nn.Module):
    """
    Lightweight frozen visual encoder that converts image observations
    into visual tokens.  Simulates a small ViT for Meta-World's simple
    visual observations.

    In a full VLA integration you would replace this with the VLA's
    own frozen vision encoder (e.g. SigLIP, DINOv2).
    """

    def __init__(self, obs_dim: int, token_dim: int, num_tokens: int = 49) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(obs_dim, token_dim * num_tokens)
        self.norm = nn.LayerNorm(token_dim)
        # Freeze
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) -> (B, num_tokens, token_dim)"""
        B = obs.shape[0]
        tokens = self.proj(obs).view(B, self.num_tokens, -1)
        return self.norm(tokens)


class SimpleLanguageEncoder(nn.Module):
    """
    Frozen language encoder that converts task names/instructions
    into language tokens.

    In a full VLA integration you would use the VLA's own frozen
    language encoder.
    """

    def __init__(self, vocab_size: int, token_dim: int, max_len: int = 10) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, token_dim) * 0.02)
        self.norm = nn.LayerNorm(token_dim)
        self.max_len = max_len
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """task_ids: (B,) integer task IDs -> (B, max_len, token_dim)"""
        emb = self.embedding(task_ids).unsqueeze(1).expand(-1, self.max_len, -1)
        return self.norm(emb + self.pos[:, :self.max_len])


# ── Dataset ───────────────────────────────────────────────────────────

class MetaWorldDemoDataset(Dataset):
    """
    In-memory dataset of Meta-World expert demonstrations.

    Each sample: (obs, task_id, action, proprio, next_action_or_none)

    If demo files don't exist, generates synthetic demos for testing.
    """

    def __init__(
        self,
        demo_dir: str | None = None,
        obs_dim: int = 39,
        action_dim: int = 4,
        proprio_dim: int = 7,
        num_tasks: int = 10,
        episodes_per_task: int = 50,
        steps_per_episode: int = 200,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim

        if demo_dir and Path(demo_dir).exists():
            self._load_demos(demo_dir)
        else:
            self._generate_synthetic(
                num_tasks, episodes_per_task, steps_per_episode,
            )

    def _generate_synthetic(
        self, num_tasks: int, episodes: int, steps: int,
    ) -> None:
        """Generate synthetic demonstrations for testing."""
        total = num_tasks * episodes * steps
        self.observations = torch.randn(total, self.obs_dim)
        self.task_ids = torch.randint(0, num_tasks, (total,))
        self.actions = torch.randn(total, self.action_dim)
        self.proprios = torch.randn(total, self.proprio_dim)

        # Next actions (shifted by 1, last step has None -> use same)
        self.next_actions = self.actions.roll(-1, dims=0)
        # Mark episode boundaries (every `steps` samples)
        self.is_first_step = torch.zeros(total, dtype=torch.bool)
        for i in range(0, total, steps):
            self.is_first_step[i] = True

    def _load_demos(self, demo_dir: str) -> None:
        """Load demonstrations from directory."""
        data = torch.load(Path(demo_dir) / "demos.pt", weights_only=True)
        self.observations = data["observations"]
        self.task_ids = data["task_ids"]
        self.actions = data["actions"]
        self.proprios = data["proprios"]
        self.next_actions = data.get("next_actions", self.actions.roll(-1, 0))
        self.is_first_step = data.get(
            "is_first_step",
            torch.zeros(len(self.observations), dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.observations[idx],
            "task_id": self.task_ids[idx],
            "action": self.actions[idx],
            "proprio": self.proprios[idx],
            "prev_action": (
                torch.zeros(self.action_dim)
                if self.is_first_step[idx]
                else self.actions[max(0, idx - 1)]
            ),
            "is_first": self.is_first_step[idx],
        }


# ── Training loop ────────────────────────────────────────────────────

def train_one_epoch(
    aegis: AEGISActionHead,
    vis_enc: SimpleVisualEncoder,
    lang_enc: SimpleLanguageEncoder,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    ema: EMAModel,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch, return average losses."""
    aegis.train()
    running: dict[str, float] = {}
    count = 0

    for batch in loader:
        obs = batch["obs"].to(device)
        task_id = batch["task_id"].to(device)
        action = batch["action"].to(device)
        proprio = batch["proprio"].to(device)
        prev_action = batch["prev_action"].to(device)
        is_first = batch["is_first"].to(device)

        with torch.no_grad():
            vis_tokens = vis_enc(obs)
            lang_tokens = lang_enc(task_id)

        output = aegis(vis_tokens, lang_tokens, proprio, gt_action=action)

        # Mask prev_action for first steps
        prev = prev_action.clone()
        prev[is_first] = action[is_first]  # no consistency penalty

        total, components = aegis_loss(output, action, prev_action=prev)

        optimiser.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(aegis.parameters(), 1.0)
        optimiser.step()
        ema.update(aegis)

        for k, v in components.items():
            running[k] = running.get(k, 0.0) + v.item()
        count += 1

    return {k: v / max(count, 1) for k, v in running.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AEGIS on Meta-World")
    parser.add_argument("--demo_dir", type=str, default=None)
    parser.add_argument("--tasks", type=str, default="MT10")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints/aegis")
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--obs_dim", type=int, default=39)
    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--proprio_dim", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    num_tasks = 10 if "10" in args.tasks else 50

    # Build dataset
    dataset = MetaWorldDemoDataset(
        demo_dir=args.demo_dir,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        num_tasks=num_tasks,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    # Build frozen encoders
    vis_enc = SimpleVisualEncoder(args.obs_dim, args.token_dim).to(device)
    lang_enc = SimpleLanguageEncoder(num_tasks + 1, args.token_dim).to(device)

    # Build AEGIS
    cfg = AEGISConfig(
        token_dim=args.token_dim,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        ien=IENConfig(
            token_dim=args.token_dim,
            action_dim=args.action_dim,
            num_heads=4,
            num_energy_layers=2,
            mlp_hidden=args.token_dim * 2,
        ),
        num_geo_queries=4,
        geo_num_heads=4,
        crit_hidden=args.token_dim,
        amort_hidden=args.token_dim * 2,
    )
    aegis = AEGISActionHead(cfg).to(device)
    ema = EMAModel(aegis, decay=0.999)

    params = aegis.param_summary()
    print("AEGIS parameter summary:")
    for k, v in params.items():
        print(f"  {k}: {format_params(v)}")

    optimiser = torch.optim.AdamW(aegis.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        losses = train_one_epoch(
            aegis, vis_enc, lang_enc, loader, optimiser, ema, epoch, device,
        )
        scheduler.step()
        dt = time.time() - t0

        log = f"Epoch {epoch+1:3d}/{args.epochs} ({dt:.1f}s) |"
        for k, v in losses.items():
            log += f" {k}={v:.4f}"
        print(log)

        if losses["total"] < best_loss:
            best_loss = losses["total"]
            torch.save(
                {"epoch": epoch, "state_dict": aegis.state_dict(), "cfg": cfg},
                save_dir / "best.pt",
            )

    # Save final + EMA
    torch.save(
        {"epoch": args.epochs, "state_dict": aegis.state_dict(), "cfg": cfg},
        save_dir / "final.pt",
    )
    ema.apply(aegis)
    torch.save(
        {"epoch": args.epochs, "state_dict": aegis.state_dict(), "cfg": cfg},
        save_dir / "ema.pt",
    )
    print(f"\nTraining complete. Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
