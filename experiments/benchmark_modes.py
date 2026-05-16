"""
Focused ablation: Mode-averaging failure of regression vs AEGIS.

Creates tasks where the expert demonstration has MULTIPLE VALID MODES.
MSE regression averages modes → invalid mean action → failure.
Energy-based sampling picks one mode → valid action → success.

This demonstrates the specific advantage of energy-based action heads.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aegis.aegis_module import AEGISActionHead, AEGISConfig
from aegis.energy_head import IENConfig
from aegis.losses import aegis_loss
from aegis.utils import format_params, EMAModel


@dataclass
class TaskSpec:
    task_id: int
    name: str
    n_modes: int
    obstacle_radius: float
    tolerance: float


TASKS = [
    TaskSpec(0, "bypass-2way",    2, 0.15, 0.08),
    TaskSpec(1, "bypass-3way",    3, 0.15, 0.08),
    TaskSpec(2, "funnel-2way",    2, 0.12, 0.06),
    TaskSpec(3, "funnel-3way",    3, 0.12, 0.06),
    TaskSpec(4, "grasp-flip",     2, 0.10, 0.07),
    TaskSpec(5, "reach-around",   2, 0.18, 0.08),
    TaskSpec(6, "slide-lr",       2, 0.14, 0.06),
    TaskSpec(7, "pivot-2way",     2, 0.13, 0.07),
    TaskSpec(8, "arc-reach",      2, 0.16, 0.08),
    TaskSpec(9, "squeeze-2way",   2, 0.10, 0.05),
]


class MultiModalEnv:
    """
    Environment with obstacle between start and goal.
    Expert goes AROUND the obstacle via one of N modes.
    MSE regression averages modes → goes THROUGH obstacle → fails.

    State: (ee[3], target[3], obstacle[3], task_one_hot[10]) = 19 dims
    Action: delta_ee[3] + grip[1] = 4 dims
    """

    OBS_DIM = 22
    ACT_DIM = 4
    PROPRIO_DIM = 7

    def __init__(self, task: TaskSpec, seed: int = 0, mode: int = -1) -> None:
        self.task = task
        self.rng = torch.Generator().manual_seed(seed)
        self.mode = mode  # -1 = random
        self.reset()

    def reset(self) -> dict[str, torch.Tensor]:
        # Start and target on opposite sides
        self.ee_pos = torch.tensor([-0.3, 0.0, 0.0])
        self.ee_pos += torch.randn(3, generator=self.rng) * 0.05
        self.target = torch.tensor([0.3, 0.0, 0.0])
        self.target += torch.randn(3, generator=self.rng) * 0.05

        # Obstacle in the middle (blocks direct path)
        self.obstacle = (self.ee_pos + self.target) / 2
        self.obstacle += torch.randn(3, generator=self.rng) * 0.02
        self.obs_radius = self.task.obstacle_radius

        # Pick mode
        if self.mode < 0:
            self.active_mode = int(
                torch.randint(self.task.n_modes, (1,), generator=self.rng).item()
            )
        else:
            self.active_mode = self.mode % self.task.n_modes

        self.ee_vel = torch.zeros(3)
        self.gripper = torch.tensor([0.0])
        self.step_count = 0
        return self._obs()

    def _obs(self) -> dict[str, torch.Tensor]:
        task_feat = torch.zeros(10)
        task_feat[self.task.task_id] = 1.0
        obs = torch.cat([
            self.ee_pos, self.target, self.obstacle, task_feat,
            torch.tensor([self.obs_radius, self.step_count / 50.0, float(self.active_mode)]),
        ])
        proprio = torch.cat([self.ee_pos, self.ee_vel, self.gripper])
        return {"obs": obs, "proprio": proprio}

    def expert_action(self) -> torch.Tensor:
        """Go around obstacle via the chosen mode."""
        to_target = self.target - self.ee_pos
        to_obs = self.obstacle - self.ee_pos
        dist_to_target = to_target.norm()
        dist_to_obs = to_obs.norm()

        # Near obstacle: deviate according to mode
        if dist_to_obs < self.obs_radius * 3:
            # Generate mode-specific bypass direction
            angle = 2 * math.pi * self.active_mode / self.task.n_modes
            # Perpendicular to obstacle direction in XY plane
            perp = torch.tensor([
                -to_obs[1] * math.cos(angle) + to_obs[2] * math.sin(angle),
                to_obs[0] * math.cos(angle),
                to_obs[0] * math.sin(angle),
            ])
            if perp.norm() > 1e-6:
                perp = perp / perp.norm()

            # Stronger bypass when closer to obstacle
            bypass_strength = max(0, 1.0 - dist_to_obs / (self.obs_radius * 3))
            action_ee = to_target / max(dist_to_target, 0.01) * 0.08 + perp * bypass_strength * 0.15
        else:
            action_ee = to_target / max(dist_to_target, 0.01) * 0.1

        # Slow down near target
        if dist_to_target < 0.15:
            action_ee = action_ee * (dist_to_target / 0.15)

        noise = torch.randn(3, generator=self.rng) * 0.008
        action_ee = action_ee + noise

        grip = torch.tensor([0.0])
        return torch.cat([action_ee, grip])

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], bool]:
        delta = action[:3].clamp(-0.2, 0.2)

        # Check collision with obstacle
        new_pos = self.ee_pos + delta
        dist_to_obs = (new_pos - self.obstacle).norm()
        if dist_to_obs < self.obs_radius * 0.5:
            # Collision: bounce back
            push_dir = new_pos - self.obstacle
            if push_dir.norm() > 1e-6:
                push_dir = push_dir / push_dir.norm()
            new_pos = self.obstacle + push_dir * self.obs_radius * 0.5
            # Penalize: reduce velocity
            delta = (new_pos - self.ee_pos) * 0.3

        self.ee_vel = delta
        self.ee_pos = self.ee_pos + delta
        self.step_count += 1

        dist = (self.ee_pos - self.target).norm().item()
        return self._obs(), dist < self.task.tolerance


def collect_demos(
    tasks: list[TaskSpec],
    episodes: int = 300,
    steps: int = 60,
) -> dict[str, torch.Tensor]:
    all_obs, all_proprio, all_action, all_tid = [], [], [], []
    all_prev, all_first = [], []

    for task in tasks:
        for ep in range(episodes):
            env = MultiModalEnv(task, seed=task.task_id * 10000 + ep)
            state = env.reset()
            prev_act = torch.zeros(4)
            for s in range(steps):
                action = env.expert_action()
                all_obs.append(state["obs"])
                all_proprio.append(state["proprio"])
                all_action.append(action)
                all_tid.append(torch.tensor(task.task_id))
                all_prev.append(prev_act.clone())
                all_first.append(torch.tensor(s == 0))
                state, _ = env.step(action)
                prev_act = action

    return {
        "obs": torch.stack(all_obs),
        "proprio": torch.stack(all_proprio),
        "action": torch.stack(all_action),
        "task_id": torch.stack(all_tid),
        "prev_action": torch.stack(all_prev),
        "is_first": torch.stack(all_first),
    }


class FrozenVisEnc(nn.Module):
    def __init__(self, obs_dim: int, d: int, n_tok: int = 16) -> None:
        super().__init__()
        self.n = n_tok
        self.proj = nn.Linear(obs_dim, d * n_tok)
        self.norm = nn.LayerNorm(d)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x).view(x.shape[0], self.n, -1))


class FrozenLangEnc(nn.Module):
    def __init__(self, n_tasks: int, d: int, seq: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_tasks, d)
        self.pos = nn.Parameter(torch.randn(1, seq, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.seq = seq
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, tid: torch.Tensor) -> torch.Tensor:
        e = self.emb(tid).unsqueeze(1).expand(-1, self.seq, -1)
        return self.norm(e + self.pos)


class BaselineMLP(nn.Module):
    def __init__(self, d: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        B = ctx.shape[0]
        q = self.pool_q.expand(B, -1, -1)
        p, _ = self.attn(q, self.norm(ctx), self.norm(ctx))
        return self.mlp(p.squeeze(1))


class GaussianMixtureHead(nn.Module):
    """Gaussian mixture model head — stronger baseline for multi-modal."""

    def __init__(self, d: int, act_dim: int, n_modes: int = 4, hidden: int = 256) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.act_dim = act_dim
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.backbone = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.means = nn.Linear(hidden, n_modes * act_dim)
        self.logits = nn.Linear(hidden, n_modes)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        B = ctx.shape[0]
        q = self.pool_q.expand(B, -1, -1)
        p, _ = self.attn(q, self.norm(ctx), self.norm(ctx))
        h = self.backbone(p.squeeze(1))
        means = self.means(h).view(B, self.n_modes, self.act_dim)
        logits = self.logits(h)
        # Pick mode with highest logit (deterministic at test time)
        idx = logits.argmax(dim=-1)  # (B,)
        return means[torch.arange(B), idx]

    def nll_loss(self, ctx: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = ctx.shape[0]
        q = self.pool_q.expand(B, -1, -1)
        p, _ = self.attn(q, self.norm(ctx), self.norm(ctx))
        h = self.backbone(p.squeeze(1))
        means = self.means(h).view(B, self.n_modes, self.act_dim)
        logits = self.logits(h)

        log_probs = F.log_softmax(logits, dim=-1)
        diff = target.unsqueeze(1) - means  # (B, K, A)
        component_ll = -0.5 * (diff ** 2).sum(-1)  # (B, K)
        mixture_ll = torch.logsumexp(log_probs + component_ll, dim=-1)
        return -mixture_ll.mean()


def train_baseline(model, vis, lang, data, epochs, bs, lr, dev):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ds = torch.utils.data.TensorDataset(data["obs"], data["task_id"], data["action"])
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(epochs):
        el, c = 0.0, 0
        model.train()
        for ob, td, ac in loader:
            ob, td, ac = ob.to(dev), td.to(dev), ac.to(dev)
            with torch.no_grad():
                v = vis(ob)
                lang_tok = lang(td)
            ctx = torch.cat([v, lang_tok], dim=1)
            loss = F.mse_loss(model(ctx), ac)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item()
            c += 1
        sched.step()
        if (ep + 1) % 25 == 0:
            print(f"  BL-MLP  ep {ep+1}/{epochs}: loss={el/c:.6f}")


def train_gmm(model, vis, lang, data, epochs, bs, lr, dev):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ds = torch.utils.data.TensorDataset(data["obs"], data["task_id"], data["action"])
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(epochs):
        el, c = 0.0, 0
        model.train()
        for ob, td, ac in loader:
            ob, td, ac = ob.to(dev), td.to(dev), ac.to(dev)
            with torch.no_grad():
                v = vis(ob)
                lang_tok = lang(td)
            ctx = torch.cat([v, lang_tok], dim=1)
            loss = model.nll_loss(ctx, ac)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item()
            c += 1
        sched.step()
        if (ep + 1) % 25 == 0:
            print(f"  BL-GMM  ep {ep+1}/{epochs}: loss={el/c:.6f}")


def train_aegis_model(model, vis, lang, data, epochs, bs, lr, dev):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ema = EMAModel(model, decay=0.999)
    ds = torch.utils.data.TensorDataset(
        data["obs"], data["proprio"], data["task_id"],
        data["action"], data["prev_action"], data["is_first"],
    )
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(epochs):
        el, c = 0.0, 0
        model.train()
        for ob, pr, td, ac, pa, fi in loader:
            ob, pr, td = ob.to(dev), pr.to(dev), td.to(dev)
            ac, pa, fi = ac.to(dev), pa.to(dev), fi.to(dev)
            with torch.no_grad():
                v = vis(ob)
                lang_tok = lang(td)
            out = model(v, lang_tok, pr, gt_action=ac)
            pv = pa.clone()
            pv[fi.bool()] = ac[fi.bool()]
            total, _ = aegis_loss(out, ac, prev_action=pv)
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            el += total.item()
            c += 1
        sched.step()
        if (ep + 1) % 25 == 0:
            print(f"  AEGIS   ep {ep+1}/{epochs}: loss={el/c:.6f}")
    ema.apply(model)


def evaluate(predict_fn, vis, lang, tasks, episodes, max_steps, dev):
    results = {}
    for task in tasks:
        succ = 0
        for ep in range(episodes):
            env = MultiModalEnv(task, seed=88888 + task.task_id * 10000 + ep)
            state = env.reset()
            ok = False
            for _ in range(max_steps):
                ob = state["obs"].unsqueeze(0).to(dev)
                pr = state["proprio"].unsqueeze(0).to(dev)
                td = torch.tensor([task.task_id], dtype=torch.long).to(dev)
                with torch.no_grad():
                    v = vis(ob)
                    lang_tok = lang(td)
                    a = predict_fn(v, lang_tok, pr)
                state, success = env.step(a.squeeze(0).cpu())
                if success:
                    ok = True
                    break
            succ += int(ok)
        results[task.name] = succ / episodes
    results["average"] = sum(v for k, v in results.items() if k != "average") / len(tasks)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes_train", type=int, default=300)
    parser.add_argument("--episodes_eval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="experiments/results_modes.json")
    args = parser.parse_args()

    dev = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {dev}")
    D = args.token_dim

    print("\n[1/7] Collecting multi-modal expert demos...")
    data = collect_demos(TASKS, args.episodes_train, steps=60)
    print(f"  {len(data['obs'])} transitions")

    torch.manual_seed(42)
    vis = FrozenVisEnc(MultiModalEnv.OBS_DIM, D).to(dev)
    lang = FrozenLangEnc(10, D).to(dev)

    # ── Train BL-MLP ──
    print("\n[2/7] Training Baseline MLP (MSE regression)...")
    torch.manual_seed(42)
    bl_mlp = BaselineMLP(D, MultiModalEnv.ACT_DIM).to(dev)
    train_baseline(bl_mlp, vis, lang, data, args.epochs, args.batch_size, args.lr, dev)

    # ── Train BL-GMM ──
    print("\n[3/7] Training Baseline GMM (mixture model)...")
    torch.manual_seed(42)
    bl_gmm = GaussianMixtureHead(D, MultiModalEnv.ACT_DIM, n_modes=4).to(dev)
    train_gmm(bl_gmm, vis, lang, data, args.epochs, args.batch_size, args.lr, dev)

    # ── Train AEGIS ──
    print("\n[4/7] Training AEGIS...")
    torch.manual_seed(42)
    cfg = AEGISConfig(
        token_dim=D, action_dim=MultiModalEnv.ACT_DIM,
        proprio_dim=MultiModalEnv.PROPRIO_DIM,
        ien=IENConfig(token_dim=D, action_dim=MultiModalEnv.ACT_DIM,
                      num_heads=4, num_energy_layers=2, mlp_hidden=D * 2),
        num_geo_queries=4, geo_num_heads=4, crit_hidden=D,
        min_refine_steps=2, max_refine_steps=8,
        langevin_lr=0.005, langevin_noise_scale=0.002,
        langevin_grad_clip=0.5,
        amort_hidden=D * 2, num_negative_samples=32,
    )
    aegis = AEGISActionHead(cfg).to(dev)
    ae_p = aegis.param_summary()
    print(f"  Params: {format_params(ae_p['total'])}")
    train_aegis_model(aegis, vis, lang, data, args.epochs, args.batch_size, args.lr, dev)

    # ── Evaluate ──
    print("\n[5/7] Evaluating BL-MLP...")
    bl_mlp.eval()
    def mlp_pred(v, lang_tok, p):
        return bl_mlp(torch.cat([v, lang_tok], dim=1))
    r_mlp = evaluate(mlp_pred, vis, lang, TASKS, args.episodes_eval, 60, dev)

    print("\n[6/7] Evaluating BL-GMM...")
    bl_gmm.eval()
    def gmm_pred(v, lang_tok, p):
        return bl_gmm(torch.cat([v, lang_tok], dim=1))
    r_gmm = evaluate(gmm_pred, vis, lang, TASKS, args.episodes_eval, 60, dev)

    print("\n[7/7] Evaluating AEGIS...")
    aegis.eval()
    def ae_pred(v, lang_tok, p):
        return aegis.predict(v, lang_tok, p)
    r_aegis = evaluate(ae_pred, vis, lang, TASKS, args.episodes_eval, 60, dev)

    # ── Print ──
    print(f"\n{'=' * 80}")
    print(f"{'Task':<18} {'Modes':<7} {'BL-MLP':<12} {'BL-GMM':<12} {'AEGIS':<12} {'Δ(MLP)':<10}")
    print("-" * 80)
    for t in TASKS:
        m = r_mlp[t.name]
        g = r_gmm[t.name]
        a = r_aegis[t.name]
        d = a - m
        s = "+" if d >= 0 else ""
        print(f"{t.name:<18} {t.n_modes:<7} {m:>8.1%}    {g:>8.1%}    {a:>8.1%}    {s}{d:>7.1%}")
    print("-" * 80)
    ma = r_mlp["average"]
    ga = r_gmm["average"]
    aa = r_aegis["average"]
    da = aa - ma
    sa = "+" if da >= 0 else ""
    print(f"{'AVERAGE':<18} {'—':<7} {ma:>8.1%}    {ga:>8.1%}    {aa:>8.1%}    {sa}{da:>7.1%}")
    print("=" * 80)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"bl_mlp": r_mlp, "bl_gmm": r_gmm, "aegis": r_aegis,
                    "delta_mlp": da, "delta_gmm": aa - ga}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
