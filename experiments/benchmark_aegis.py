"""
Controlled ablation: Baseline regression MLP vs AEGIS action heads.

Creates a challenging multi-task manipulation benchmark with:
  - Multi-modal expert strategies (multiple valid solutions)
  - Contact-transition dynamics (phase switches)
  - Distribution shift at evaluation (shifted goals)
  - Non-linear dynamics with coupling

These properties stress-test exactly the capabilities AEGIS is
designed for: energy-based multi-modality, criticality-aware
refinement, and geometric grounding.
"""

from __future__ import annotations

import argparse
import json

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

# ── Task definitions ──────────────────────────────────────────────────

TASKS = {
    0: {"name": "reach-easy",     "noise": 0.02, "tol": 0.12, "phase": "free",      "multimodal": False},
    1: {"name": "reach-hard",     "noise": 0.05, "tol": 0.06, "phase": "free",      "multimodal": False},
    2: {"name": "push-straight",  "noise": 0.04, "tol": 0.08, "phase": "contact",   "multimodal": False},
    3: {"name": "push-around",    "noise": 0.05, "tol": 0.07, "phase": "contact",   "multimodal": True},
    4: {"name": "pick-place",     "noise": 0.06, "tol": 0.06, "phase": "contact",   "multimodal": True},
    5: {"name": "door-open",      "noise": 0.05, "tol": 0.07, "phase": "contact",   "multimodal": False},
    6: {"name": "insert-peg",     "noise": 0.08, "tol": 0.04, "phase": "precision",  "multimodal": False},
    7: {"name": "rotate-handle",  "noise": 0.07, "tol": 0.04, "phase": "precision",  "multimodal": True},
    8: {"name": "stack-blocks",   "noise": 0.09, "tol": 0.03, "phase": "precision",  "multimodal": True},
    9: {"name": "place-cup",      "noise": 0.07, "tol": 0.04, "phase": "precision",  "multimodal": True},
}


@dataclass
class TaskSpec:
    task_id: int
    name: str
    noise: float
    tolerance: float
    phase: str
    multimodal: bool


def get_task_specs(n: int = 10) -> list[TaskSpec]:
    return [
        TaskSpec(i, TASKS[i]["name"], TASKS[i]["noise"], TASKS[i]["tol"],
                 TASKS[i]["phase"], TASKS[i]["multimodal"])
        for i in range(min(n, len(TASKS)))
    ]


# ── Environment ───────────────────────────────────────────────────────

class ManipulationEnv:
    """
    Multi-task manipulation with challenging dynamics.

    Key properties that stress-test action heads:
    1. Non-linear dynamics with friction and coupling
    2. Multi-modal expert policies (multiple valid strategies)
    3. Phase-dependent dynamics (free-space vs contact)
    4. High-dimensional observation with partial observability
    """

    OBS_DIM = 39  # 3+3+3+30
    ACT_DIM = 4
    PROPRIO_DIM = 7

    def __init__(self, task: TaskSpec, seed: int = 0, shift: float = 0.0) -> None:
        self.task = task
        self.rng = torch.Generator().manual_seed(seed)
        self.shift = shift  # distribution shift for eval
        self.reset()

    def reset(self) -> dict[str, torch.Tensor]:
        self.ee_pos = torch.randn(3, generator=self.rng) * 0.2
        self.ee_vel = torch.zeros(3)
        self.gripper = torch.tensor([0.0])
        self.obj_pos = torch.randn(3, generator=self.rng) * 0.2

        # Shifted target at eval
        self.target_pos = torch.randn(3, generator=self.rng) * 0.2
        if self.shift > 0:
            self.target_pos += torch.randn(3, generator=self.rng) * self.shift

        self.in_contact = False
        self.step_count = 0
        return self._obs()

    def _obs(self) -> dict[str, torch.Tensor]:
        # Partial observability: add noise to obj/target positions
        obs_noise = torch.randn(6, generator=self.rng) * 0.01
        obj_obs = self.obj_pos + obs_noise[:3]
        tgt_obs = self.target_pos + obs_noise[3:]

        # Task encoding
        task_feat = torch.zeros(30)
        task_feat[self.task.task_id] = 1.0
        phase_map = {"free": 0, "contact": 1, "precision": 2}
        task_feat[10 + phase_map[self.task.phase]] = 1.0
        task_feat[13] = self.task.noise
        task_feat[14] = self.task.tolerance
        task_feat[15] = float(self.in_contact)
        task_feat[16] = self.step_count / 50.0

        obs = torch.cat([self.ee_pos, obj_obs, tgt_obs, task_feat])  # 39
        proprio = torch.cat([self.ee_pos, self.ee_vel, self.gripper])  # 7
        return {"obs": obs, "proprio": proprio}

    def _dynamics(self, action: torch.Tensor) -> None:
        """Non-linear dynamics with friction and phase transitions."""
        delta_ee = action[:3]
        grip = action[3:4]

        # Friction (stronger near objects and in contact)
        dist_to_obj = (self.ee_pos - self.obj_pos).norm()
        friction = 0.1 + 0.3 * torch.exp(-dist_to_obj * 5)

        # Non-linear damping
        damped = delta_ee * (1.0 - friction)

        # Coupling: gripper state affects movement precision
        if self.task.phase != "free":
            grip_effect = 1.0 - 0.3 * grip.item()
            damped = damped * grip_effect

        # Contact dynamics
        if dist_to_obj < 0.1:
            self.in_contact = True
            # Object moves with end-effector in contact
            self.obj_pos = self.obj_pos + damped * 0.5
        else:
            self.in_contact = False

        self.ee_vel = damped
        self.ee_pos = self.ee_pos + damped
        self.gripper = grip.clamp(0, 1)
        self.step_count += 1

    def expert_action(self) -> torch.Tensor:
        """Multi-modal expert with task-dependent strategies."""
        direction = self.target_pos - self.ee_pos
        dist = direction.norm()

        if self.task.multimodal:
            # Two equally valid strategies
            coin = torch.rand(1, generator=self.rng).item()
            if coin < 0.5:
                # Strategy A: direct approach
                gain = 0.4
                action_ee = direction * gain
            else:
                # Strategy B: curved approach (go via intermediate point)
                perp = torch.tensor([direction[1], -direction[0], direction[2]])
                if perp.norm() > 1e-6:
                    perp = perp / perp.norm()
                curve_strength = 0.15 * max(0, 1.0 - self.step_count / 25.0)
                action_ee = direction * 0.35 + perp * curve_strength
        else:
            gain = 0.35
            action_ee = direction * gain

        # Phase-dependent precision
        if self.task.phase == "precision" and dist < 0.15:
            action_ee = action_ee * 0.5  # slow down near target

        # Task-dependent noise
        noise = torch.randn(3, generator=self.rng) * self.task.noise
        action_ee = action_ee + noise

        # Gripper action
        if self.task.phase == "free":
            grip = torch.tensor([0.0])
        elif self.in_contact or dist < 0.15:
            grip = torch.tensor([1.0])
        else:
            grip = torch.tensor([0.0])

        return torch.cat([action_ee, grip])

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], bool]:
        self._dynamics(action)
        dist = (self.ee_pos - self.target_pos).norm().item()
        success = dist < self.task.tolerance
        return self._obs(), success


# ── Data collection ───────────────────────────────────────────────────

def collect_demos(
    tasks: list[TaskSpec],
    episodes: int = 200,
    steps: int = 50,
) -> dict[str, torch.Tensor]:
    all_obs, all_proprio, all_action, all_tid = [], [], [], []
    all_prev, all_first = [], []

    for task in tasks:
        for ep in range(episodes):
            env = ManipulationEnv(task, seed=task.task_id * 10000 + ep)
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


# ── Frozen encoders ───────────────────────────────────────────────────

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


# ── Baseline MLP ──────────────────────────────────────────────────────

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


# ── Training routines ─────────────────────────────────────────────────

def train_baseline(
    model: BaselineMLP,
    vis: FrozenVisEnc,
    lang: FrozenLangEnc,
    data: dict[str, torch.Tensor],
    epochs: int,
    bs: int,
    lr: float,
    dev: torch.device,
) -> list[float]:
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ds = torch.utils.data.TensorDataset(data["obs"], data["task_id"], data["action"])
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    losses = []
    for ep in range(epochs):
        el = 0.0
        c = 0
        model.train()
        for ob, td, ac in loader:
            ob, td, ac = ob.to(dev), td.to(dev), ac.to(dev)
            with torch.no_grad():
                v = vis(ob)
                lang_tok = lang(td)
            ctx = torch.cat([v, lang_tok], dim=1)
            pred = model(ctx)
            loss = F.mse_loss(pred, ac)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item()
            c += 1
        sched.step()
        avg = el / max(c, 1)
        losses.append(avg)
        if (ep + 1) % 20 == 0:
            print(f"  Baseline ep {ep+1}/{epochs}: loss={avg:.6f}")
    return losses


def train_aegis_model(
    model: AEGISActionHead,
    vis: FrozenVisEnc,
    lang: FrozenLangEnc,
    data: dict[str, torch.Tensor],
    epochs: int,
    bs: int,
    lr: float,
    dev: torch.device,
) -> list[float]:
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ema = EMAModel(model, decay=0.999)
    ds = torch.utils.data.TensorDataset(
        data["obs"], data["proprio"], data["task_id"],
        data["action"], data["prev_action"], data["is_first"],
    )
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    losses = []
    for ep in range(epochs):
        el = 0.0
        c = 0
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
        avg = el / max(c, 1)
        losses.append(avg)
        if (ep + 1) % 20 == 0:
            print(f"  AEGIS   ep {ep+1}/{epochs}: loss={avg:.6f}")
    ema.apply(model)
    return losses


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(
    predict_fn,
    vis: FrozenVisEnc,
    lang: FrozenLangEnc,
    tasks: list[TaskSpec],
    episodes: int = 200,
    max_steps: int = 50,
    shift: float = 0.0,
    dev: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    results: dict[str, float] = {}
    for task in tasks:
        succ = 0
        for ep in range(episodes):
            env = ManipulationEnv(
                task, seed=77777 + task.task_id * 10000 + ep, shift=shift,
            )
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


# ── Main ──────────────────────────────────────────────────────────────

def print_table(
    tasks: list[TaskSpec],
    bl: dict[str, float],
    ae: dict[str, float],
    label: str,
) -> None:
    print(f"\n{'=' * 76}")
    print(f"  {label}")
    print(f"{'=' * 76}")
    print(f"{'Task':<20} {'Phase':<12} {'Baseline':<12} {'AEGIS':<12} {'Delta':<10}")
    print("-" * 76)
    for t in tasks:
        b = bl[t.name]
        a = ae[t.name]
        d = a - b
        s = "+" if d >= 0 else ""
        print(f"{t.name:<20} {t.phase:<12} {b:>8.1%}    {a:>8.1%}    {s}{d:>7.1%}")
    print("-" * 76)
    ba, aa = bl["average"], ae["average"]
    da = aa - ba
    sa = "+" if da >= 0 else ""
    print(f"{'AVERAGE':<20} {'all':<12} {ba:>8.1%}    {aa:>8.1%}    {sa}{da:>7.1%}")
    print("=" * 76)

    # Phase breakdown
    for phase in ["free", "contact", "precision"]:
        pt = [t for t in tasks if t.phase == phase]
        bp = sum(bl[t.name] for t in pt) / len(pt)
        ap = sum(ae[t.name] for t in pt) / len(pt)
        dp = ap - bp
        sp = "+" if dp >= 0 else ""
        print(f"  {phase:<12} {bp:>8.1%} -> {ap:>8.1%}  ({sp}{dp:.1%})")

    # Multimodal vs unimodal
    mm = [t for t in tasks if t.multimodal]
    um = [t for t in tasks if not t.multimodal]
    if mm:
        bm = sum(bl[t.name] for t in mm) / len(mm)
        am = sum(ae[t.name] for t in mm) / len(mm)
        dm = am - bm
        sm = "+" if dm >= 0 else ""
        print(f"  multimodal   {bm:>8.1%} -> {am:>8.1%}  ({sm}{dm:.1%})")
    if um:
        bu = sum(bl[t.name] for t in um) / len(um)
        au = sum(ae[t.name] for t in um) / len(um)
        du = au - bu
        su = "+" if du >= 0 else ""
        print(f"  unimodal     {bu:>8.1%} -> {au:>8.1%}  ({su}{du:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes_train", type=int, default=200)
    parser.add_argument("--episodes_eval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="experiments/results.json")
    args = parser.parse_args()

    dev = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {dev}")
    D = args.token_dim
    tasks = get_task_specs(10)

    # ── Data ──
    print("\n[1/6] Collecting expert demonstrations...")
    data = collect_demos(tasks, args.episodes_train, steps=50)
    print(f"  {len(data['obs'])} transitions")

    # ── Shared frozen encoders ──
    torch.manual_seed(42)
    vis = FrozenVisEnc(ManipulationEnv.OBS_DIM, D).to(dev)
    lang = FrozenLangEnc(10, D).to(dev)

    # ── Train baseline ──
    print("\n[2/6] Training Baseline MLP...")
    torch.manual_seed(42)
    baseline = BaselineMLP(D, ManipulationEnv.ACT_DIM, hidden=256).to(dev)
    bl_p = sum(p.numel() for p in baseline.parameters())
    print(f"  Params: {format_params(bl_p)}")
    train_baseline(baseline, vis, lang, data, args.epochs, args.batch_size, args.lr, dev)

    # ── Train AEGIS ──
    print("\n[3/6] Training AEGIS...")
    torch.manual_seed(42)
    cfg = AEGISConfig(
        token_dim=D, action_dim=ManipulationEnv.ACT_DIM,
        proprio_dim=ManipulationEnv.PROPRIO_DIM,
        ien=IENConfig(token_dim=D, action_dim=ManipulationEnv.ACT_DIM,
                      num_heads=4, num_energy_layers=2, mlp_hidden=D * 2),
        num_geo_queries=4, geo_num_heads=4, crit_hidden=D,
        min_refine_steps=1, max_refine_steps=8,
        amort_hidden=D * 2, num_negative_samples=32,
    )
    aegis = AEGISActionHead(cfg).to(dev)
    ae_p = aegis.param_summary()
    print(f"  Params: {format_params(ae_p['total'])}")
    train_aegis_model(aegis, vis, lang, data, args.epochs, args.batch_size, args.lr, dev)

    # ── Evaluate: in-distribution ──
    print("\n[4/6] Evaluating in-distribution (shift=0.0)...")
    baseline.eval()
    aegis.eval()

    def bl_pred(v: torch.Tensor, lang_tok: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return baseline(torch.cat([v, lang_tok], dim=1))

    def ae_pred(v: torch.Tensor, lang_tok: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return aegis.predict(v, lang_tok, p)

    bl_id = evaluate(bl_pred, vis, lang, tasks, args.episodes_eval, shift=0.0, dev=dev)
    ae_id = evaluate(ae_pred, vis, lang, tasks, args.episodes_eval, shift=0.0, dev=dev)
    print_table(tasks, bl_id, ae_id, "IN-DISTRIBUTION (shift=0.0)")

    # ── Evaluate: mild distribution shift ──
    print("\n[5/6] Evaluating with mild shift (shift=0.1)...")
    bl_s1 = evaluate(bl_pred, vis, lang, tasks, args.episodes_eval, shift=0.1, dev=dev)
    ae_s1 = evaluate(ae_pred, vis, lang, tasks, args.episodes_eval, shift=0.1, dev=dev)
    print_table(tasks, bl_s1, ae_s1, "MILD SHIFT (shift=0.1)")

    # ── Evaluate: strong distribution shift ──
    print("\n[6/6] Evaluating with strong shift (shift=0.2)...")
    bl_s2 = evaluate(bl_pred, vis, lang, tasks, args.episodes_eval, shift=0.2, dev=dev)
    ae_s2 = evaluate(ae_pred, vis, lang, tasks, args.episodes_eval, shift=0.2, dev=dev)
    print_table(tasks, bl_s2, ae_s2, "STRONG SHIFT (shift=0.2)")

    # ── Save ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "in_distribution": {"baseline": bl_id, "aegis": ae_id,
                            "delta": ae_id["average"] - bl_id["average"]},
        "mild_shift": {"baseline": bl_s1, "aegis": ae_s1,
                       "delta": ae_s1["average"] - bl_s1["average"]},
        "strong_shift": {"baseline": bl_s2, "aegis": ae_s2,
                         "delta": ae_s2["average"] - bl_s2["average"]},
        "params": {"baseline": bl_p, "aegis": ae_p},
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
