# AEGIS-VLA Novelty Analysis — Systematic Literature Comparison

## Methodology

Exhaustive search across arXiv, OpenReview, HuggingFace Papers, and Google Scholar (May 2026) for all concurrent/prior work on:
1. Energy-based action heads in VLAs
2. Adaptive test-time computation in VLAs
3. Geometric/spatial tokens with proprioceptive conditioning in VLAs

---

## 1. Energy-Based Action Heads for VLAs

| Paper | Venue | What it does | Difference from AEGIS |
|-------|-------|-------------|----------------------|
| **EBT-Policy** (Huang et al., Oct 2025) | arXiv | Standalone energy-based transformer policy; not integrated into VLA pipeline; no visual-language conditioning | AEGIS integrates energy head INTO a VLA with action-conditioned cross-attention to visual-language context. EBT-Policy is a standalone policy model, not a VLA module. |
| **Energy Policy** (2025) | arXiv | EBM for multimodal actions in single forward pass | Standalone policy; no VLA integration; no criticality-aware compute |
| **Energy-Based Action Heads Know When They Don't Know** (May 2026) | OpenReview (under review) | Energy-based action heads for OOD detection | Focuses on uncertainty/OOD detection, not action generation quality. Does not include criticality-aware adaptive refinement or geometric tokens |
| **Diffusion Policy** (Chi et al., 2023) | RSS | Iterative denoising for action generation | Fixed 100 denoising steps; no adaptive compute; diffusion (not energy) based |
| **π₀** (Black et al., 2024) | arXiv | Flow matching action generation | Fixed denoising schedule; no energy landscape; no adaptive compute |

**AEGIS novelty on C1:** First VLA module combining (a) energy-based scoring with action-conditioned cross-attention to VL context, (b) spectral normalisation for Lipschitz stability, (c) InfoNCE contrastive training, and (d) Langevin sampling from an amortised warm-start. The amortised warm-start + energy refinement architecture is distinct from EBT-Policy's direct EBM and from diffusion/flow-matching approaches.

---

## 2. Adaptive Test-Time Computation in VLAs

| Paper | Venue | What it does | Difference from AEGIS |
|-------|-------|-------------|----------------------|
| **RD-VLA** (Tur et al., Feb 2026) | ES-Reasoning@ICLR 2026 | Weight-tied recurrent action head with adaptive stopping based on **latent convergence** (L2 distance between consecutive latent states) | AEGIS uses **energy gradient magnitude** as criticality signal, not latent convergence. Different mechanism: Langevin dynamics vs recurrent blocks. Self-supervised from energy landscape geometry. |
| **VLA-ATTC** (Li et al., May 2026) | ICML 2026 | Uncertainty-based "cognitive clutch" + Relative Action Critic for candidate selection | Uses ensemble uncertainty for triggering; critic evaluates candidates pairwise. AEGIS uses energy gradient magnitude (scalar, no ensembles). Different inference: Langevin refinement vs candidate reranking. |
| **AC²-VLA** (Yu et al., Jan 2026) | arXiv | Action-context routing for token pruning + layer skipping + cache reuse (efficiency focus) | Efficiency optimization (speedup), not action quality improvement. Does not refine actions iteratively. |
| **CoT-VLA** (2025) | arXiv | Chain-of-thought visual reasoning before action | Explicit token generation; linear memory scaling; not iterative refinement |

**AEGIS novelty on C2:** The criticality signal is derived from the **geometry of the energy landscape** (gradient magnitude at GT action), which is:
- Self-supervised (no manual phase labels)
- Derived from the action head's own learned representation
- Physically interpretable (steep landscape = action sensitivity = need more refinement)
- None of RD-VLA (latent convergence), VLA-ATTC (ensemble uncertainty), or AC²-VLA (routing) use this signal

---

## 3. Geometric/Spatial Tokens with Proprioceptive Conditioning

| Paper | Venue | What it does | Difference from AEGIS |
|-------|-------|-------------|----------------------|
| **GST-VLA** (Sarowar et al., Mar 2026) | arXiv | 3D Gaussian Spatial Tokens from **dense depth maps** | Requires depth sensor; uses Gaussian primitives. AEGIS GIT uses learned cross-attention queries conditioned on proprioception, no depth needed. |
| **ThinkProprio** (Wang et al., Feb 2026) | arXiv | Proprioception tokenised as text tokens for early fusion | Converts proprio to text tokens in VLM space; doesn't extract geometric interaction descriptors. AEGIS GIT produces compact geometric tokens via cross-attention. |
| **SG-VLA** (Tu et al., 2026) | arXiv | Spatial grounding for mobile manipulation | 2D spatial grounding via waypoint prediction; not geometric interaction tokens |
| **FALCON** (2025) | arXiv | 3D point cloud grounding | Requires 3D point clouds from depth cameras |
| **Avi** (2025) | arXiv | Affordance-centric 3D grounding | Requires depth; affordance-specific |

**AEGIS novelty on C3:** GIT extracts geometric interaction descriptors from 2D visual tokens ONLY, conditioned on proprioception via learned cross-attention queries. This is distinct from:
- GST-VLA: requires depth sensor input
- ThinkProprio: text tokenisation of proprio (no geometric extraction)
- FALCON/Avi: require 3D point clouds

---

## 4. Combined Novelty

No existing work combines all three:

| | Energy action head | Adaptive refinement | Geometric tokens | All three |
|---|:---:|:---:|:---:|:---:|
| EBT-Policy | ✓ | ✗ | ✗ | ✗ |
| RD-VLA | ✗ | ✓ | ✗ | ✗ |
| VLA-ATTC | ✗ | ✓ | ✗ | ✗ |
| GST-VLA | ✗ | ✗ | ✓ | ✗ |
| ThinkProprio | ✗ | ✗ | partial | ✗ |
| Diffusion Policy | ✗ | fixed | ✗ | ✗ |
| **AEGIS-VLA** | **✓** | **✓** | **✓** | **✓** |

The synergy between components is itself novel:
- Energy landscape geometry → criticality signal → adaptive refinement steps
- Geometric tokens → enriched context → better energy landscape → better actions
- Criticality → concentrated compute on precision phases → SR improvement

---

## Summary of Novelty Claims

1. **C1 (IEN):** First energy-based action head integrated into a VLA with action-conditioned cross-attention, spectral normalisation, and amortised warm-start + Langevin refinement. Differs from EBT-Policy (standalone, no VLA integration) and from "Energy-Based Action Heads Know When They Don't Know" (focuses on OOD detection, not action quality).

2. **C2 (CAAC):** First use of energy gradient magnitude as a self-supervised criticality signal for adaptive computation in VLAs. Differs from RD-VLA (latent convergence), VLA-ATTC (ensemble uncertainty), and AC²-VLA (efficiency routing).

3. **C3 (GIT):** First proprioception-conditioned cross-attention for geometric token extraction from 2D visual tokens without depth sensors. Differs from GST-VLA (requires depth), ThinkProprio (text tokenisation), and FALCON/Avi (require 3D point clouds).

4. **Combined framework:** No prior work unifies energy-based action generation, criticality-aware adaptive computation, and geometric interaction tokens.
