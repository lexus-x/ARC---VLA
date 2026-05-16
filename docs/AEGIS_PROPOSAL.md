# AEGIS-VLA: Adaptive Energy-Guided Interaction Synthesis for Vision-Language-Action Models

## Abstract

We introduce AEGIS-VLA, a novel action-generation module for Vision-Language-Action models that replaces standard regression or diffusion-based action heads with energy-based action generation featuring three key innovations: (1) an **Interaction Energy Network** that learns a conditional energy landscape over the action space, enabling multi-modal action generation via Langevin dynamics; (2) a **Criticality-Aware Adaptive Computation** mechanism that dynamically allocates refinement steps based on a self-supervised estimate of per-timestep manipulation difficulty; and (3) **Geometric Interaction Tokens** that extract compact spatial descriptors encoding end-effector-to-object geometry, providing explicit 3D grounding from 2D visual inputs alone. AEGIS is plug-and-play: it wraps any frozen VLA backbone and is trained end-to-end with a composite loss comprising contrastive energy, amortised action prediction, self-supervised criticality, and temporal consistency objectives. We target 5-10% absolute success-rate improvement over current SOTA on Meta-World (MT10/MT50) and LIBERO benchmarks.

---

## 1. Motivation and Gap Analysis

### 1.1 Current VLA Action Heads and Their Limitations

| Action Head Type | Examples | Limitation |
|---|---|---|
| Discrete tokens (autoregressive) | RT-2, OpenVLA | Jerky single-step; quantisation loss |
| Flow matching (iterative denoising) | π0, π0.5 | Fixed compute; 50-100 denoising steps regardless of difficulty |
| Action chunking (regression) | ACT, SmolVLA | Open-loop; no reactivity within chunk |
| Text-as-action | VLA-0 | Loses continuous precision |

**None** of these adapt their computational budget to the difficulty of the current manipulation phase.

### 1.2 Three Unaddressed Gaps

1. **Uniform compute allocation.** All timesteps receive identical processing, yet free-space reaching is trivially easy while contact transitions require multi-modal precision.

2. **No energy-based VLA action heads exist.** EBT-Policy and Energy Policy demonstrate energy-based superiority over diffusion in standalone policy learning, but have never been integrated into a VLA architecture with visual-language conditioning.

3. **Geometric grounding deficit.** Standard VLAs process 2D visual tokens with no explicit 3D spatial reasoning, causing imprecise contact actions.

---

## 2. Technical Contributions

### C1: Interaction Energy Network (IEN)

A conditional energy function E(a | o, l) : R^A → R that scores candidate actions against fused visual-language context via action-conditioned cross-attention with spectral normalisation for Lipschitz stability.

**Training:** InfoNCE contrastive loss — GT actions receive low energy, Gaussian-perturbed negatives receive high energy.

**Inference:** Langevin MCMC from an amortised warm-start.

### C2: Criticality-Aware Adaptive Computation (CAAC)

A lightweight estimator predicts criticality c ∈ [0,1] from visual tokens + proprioception. The number of Langevin refinement steps is N = 1 + round(c × (N_max - 1)).

**Self-supervised target:** c* = σ(α · ‖∂E/∂a‖₂) — high energy-gradient magnitude at the GT action indicates a steep landscape requiring careful refinement.

**Novel property:** This is the first application of adaptive test-time compute to VLA action generation. Unlike early-exit in NLP (which skips layers), CAAC controls iterative refinement depth.

### C3: Geometric Interaction Tokens (GIT)

G learned queries (G=4) cross-attend to visual tokens, conditioned on proprioceptive state. This produces G geometric tokens encoding end-effector-to-object spatial relationships.

**Key advantage:** No depth sensor or point cloud required — geometric structure is extracted from the VLA's existing 2D visual tokens via learned cross-attention, conditioned on the robot's joint state.

---

## 3. Architecture

```
Visual Tokens (B, N, D) ──────────────┐
                                       │
Language Tokens (B, M, D) ────────────┤
                                       │
Proprioception (B, P) ───┬────────────┤
                          │            │
                    ┌─────▼─────┐      │
                    │   GIT     │      │
                    │ (G queries)│      │
                    └─────┬─────┘      │
                          │            │
                    G geo tokens       │
                          │            │
              Context = [vis; lang; geo]
                          │
              ┌───────────┤
              │           │
    ┌─────────▼──────┐    │
    │  Criticality   │    │
    │  Estimator     │    │
    └────────┬───────┘    │
             │            │
         c ∈ [0,1]        │
             │            │
    ┌────────▼────────┐   │
    │  Amortised      │   │
    │  Predictor      │◄──┘
    └────────┬────────┘
             │
         a_init (warm-start)
             │
    ┌────────▼────────┐
    │  Langevin       │
    │  Refinement     │  N(c) steps
    │  on E(a|o,l)    │
    └────────┬────────┘
             │
         a_final (refined action)
```

---

## 4. Loss Function

L = λ₁ · L_energy + λ₂ · L_amort + λ₃ · L_crit + λ₄ · L_consist

| Term | Formula | Purpose |
|---|---|---|
| L_energy | InfoNCE(E(a*), E(a⁻)) | Shape the energy landscape |
| L_amort | MSE(â_init, a*) | Warm-start quality |
| L_crit | BCE(ĉ, σ(α·‖∇ₐE‖)) | Self-supervised criticality |
| L_consist | MSE(â_t, â_{t-1}) | Temporal smoothness |

Default weights: λ₁=1.0, λ₂=1.0, λ₃=0.5, λ₄=0.1.

---

## 5. Why 5-10% SR Improvement Is Expected

| Mechanism | Expected Gain | Reasoning |
|---|---|---|
| Energy-based multi-modality | +2-4% | Handles multiple valid strategies (EBT-Policy shows consistent gains over diffusion) |
| Adaptive compute | +2-3% | Concentrates capacity on contact transitions where most failures occur |
| Geometric tokens | +1-3% | Explicit spatial grounding reduces positioning errors |
| Reduced multi-task interference | +1-2% | Energy landscape naturally separates task strategies |

**Evidence base:**
- EBT-Policy: 50× fewer inference steps than Diffusion Policy with equal/better SR
- Energy Policy: SOTA on MimicGen with single forward pass
- EquAct: SE(3) equivariance yields +5-15% on RLBench via geometric structure
- MoLe-VLA: Adaptive layer selection shows heterogeneous computation helps

---

## 6. Experimental Plan

### 6.1 Benchmarks

| Benchmark | Tasks | Metric | Current SOTA |
|---|---|---|---|
| Meta-World MT10 | 10 tasks | Avg SR (%) | ~85% (multi-task RL) |
| Meta-World MT50 | 50 tasks | Avg SR (%) | ~70% (multi-task RL) |
| LIBERO-Spatial | 10 tasks | Avg SR (%) | ~96% |
| LIBERO-Long | 10 tasks | Avg SR (%) | ~92% |

### 6.2 Baselines

- OpenVLA / OpenVLA-OFT (regression head)
- π0 (flow matching head)
- SmolVLA (action chunking head)
- Diffusion Policy (diffusion head)
- EBT-Policy (energy head, non-VLA)

### 6.3 Ablations

| Ablation | Tests |
|---|---|
| IEN only (no CAAC, no GIT) | Energy head contribution |
| IEN + GIT (no CAAC) | Geometric token contribution |
| IEN + CAAC (no GIT) | Adaptive compute contribution |
| Fixed N steps vs adaptive | CAAC benefit |
| G = {1, 2, 4, 8} queries | GIT capacity |
| N_max = {2, 4, 8, 16} | Refinement budget |

---

## 7. Novelty Claims

1. **First energy-based action head integrated into a VLA architecture** with visual-language conditioning via action-conditioned cross-attention.

2. **First criticality-aware adaptive computation for VLA action generation**, with a self-supervised training signal derived from energy landscape geometry.

3. **First geometric interaction tokens** that extract 3D spatial structure from 2D visual tokens via proprioception-conditioned cross-attention, without requiring depth sensors.

---

## 8. Target Venues

| Venue | Tier | Fit |
|---|---|---|
| IEEE T-RO | Q1 | Manipulation + systems focus |
| IJRR | Q1 | Foundational robotics |
| Science Robotics | Q1 | High-impact robotics |
| RA-L | Q1/Q2 | Rapid publication, robotics automation |
| IEEE TPAMI | Q1 | If emphasising the ML contribution |
