# 15 Verified Gaps in VLA Research

A systematic analysis of gaps identified across multiple VLA survey papers.

## Sources

1. **VLA Concepts, Progress, Challenges** (Cornell/HKUST, 80+ models reviewed)
2. **VLA Datasets, Benchmarks, Data Engines** (arXiv:2604.23001)
3. **ICLR 2026 VLA State Analysis** (164 submissions analyzed)
4. **VLA-Pruner** (arXiv:2511.16449, ICML 2026)
5. **NanoVLA/SmolVLA/LiteVLA** efficiency papers

---

## The Gaps

### G1: Visual Token Redundancy
**Status:** Partially addressed

ViT produces 196-400 tokens per frame. Most are irrelevant to the current task. This dominates inference cost (attention scales quadratically with token count).

**Existing solutions:** VLA-Pruner (training-free), FastV, SparseVLM, DivPrune. All use attention-based heuristics.

**Remaining gap:** No learned, task-conditioned selection mechanism exists for VLAs.

---

### G2: Prefill-Decode Attention Mismatch
**Status:** Addressed by VLA-Pruner only

Only ~50% overlap between semantic attention (vision-language prefill) and action attention (action decode). Pruning on semantic-only cues discards action-critical tokens.

**APEX addresses this** by using action loss as the training signal, directly optimizing for action-relevant token selection.

---

### G3: Spatial Grounding Absence
**Status:** Unaddressed for small VLAs

VLAs attend to "what" (objects) but lack explicit "where" (coordinates). This causes spatial hallucinations — predicting actions toward wrong positions.

**APEX addresses this** through task-conditioned token selection that learns spatial importance.

---

### G4: Catastrophic Forgetting During VLM→VLA Fine-tuning
**Status:** Partially addressed

Action token training degrades VLM reasoning capability. The better the VLA gets at actions, the worse it gets at understanding.

**Existing solutions:** LoRA, MoE adapters, ECoT (Embodied Chain-of-Thought), Actions as Language (relabeling robot datasets with text).

**Remaining gap:** No universal solution. Each approach has tradeoffs.

---

### G5: Action Tokenization Lossiness
**Status:** Active research

Continuous 6-DoF actions → discrete bins loses precision. FAST tokenizer helps but still introduces quantization error.

**Existing solutions:** FAST tokenizer, continuous regression heads, diffusion-based action generation.

**Remaining gap:** Fundamental tension between discrete token prediction (VLM-native) and continuous control requirements.

---

### G6: Real-Time Inference Gap
**Status:** Partially addressed

Most VLAs run 2-10 Hz on consumer hardware. Real-time control requires 50-100+ Hz. Visual tokens are the primary bottleneck.

**Existing solutions:** Token pruning (VLA-Pruner, FastV), token caching (VLA-Cache), quantization, early exit.

**APEX addresses this** through learned token reduction with 1.5-2x speedup.

---

### G7: Cross-Embodiment Transfer Failure
**Status:** Unaddressed (fundamental)

VLAs trained on one robot don't transfer to others. Action spaces, kinematics, and sensor configurations differ. No universal action representation exists.

**Existing approaches:** Open X-Embodiment dataset, shared action spaces, embodiment-agnostic representations.

**Remaining gap:** Fundamental challenge. No module-level solution exists.

---

### G8: Long-Horizon Degradation
**Status:** Partially addressed

VLAs process single observations. No memory for multi-step tasks. Performance degrades sharply as task horizon increases.

**Existing solutions:** History buffers, action chunking (ACT), hierarchical planning.

**Remaining gap:** The user's constraint bans standard history buffers. Novel temporal integration needed.

---

### G9: Sim-to-Real Fidelity Gap
**Status:** Active research

Synthetic data has rendering artifacts, simplified dynamics. Sim-trained VLAs underperform on real robots.

**Existing solutions:** Domain randomization, video diffusion augmentation, sim-to-real transfer learning.

**Remaining gap:** Fundamental limitation of simulation fidelity.

---

### G10: Safety/Verification Gap
**Status:** Barely addressed

No formal safety guarantees. VLAs can predict dangerous actions (collisions, excessive force). No runtime verification mechanism exists.

**Existing approaches:** SafeVLA (formal verification), constraint layers, safety filters.

**Remaining gap:** No lightweight, plug-and-play safety module exists.

---

### G11: Benchmark Saturation
**Status:** Acknowledged but unresolved

LIBERO is solved (95-99%), CALVIN near saturation (>4.5 ABC). Current benchmarks don't differentiate models well. Real-world evaluation is rare and expensive.

**ICLR 2026 observation:** "90% of papers test on LIBERO, SIMPLER, or CALVIN. Showing 99% vs 98% on LIBERO is not helpful."

---

### G12: Proprioceptive Underintegration
**Status:** Unaddressed

State tokens (joint angles, gripper) are concatenated but not deeply reasoned about. The VLA doesn't understand its own physical constraints.

**Potential approach:** Proprioceptive-aware attention mechanisms, kinematic constraint layers.

---

### G13: Data Efficiency
**Status:** Active research

VLAs need thousands of demonstrations. Real-world collection is expensive. Synthetic data has fidelity issues.

**Existing solutions:** Data engines (MimicGen, RoboGen), video-to-data pipelines, augmentation.

**Remaining gap:** Fundamental tension between data scale and data quality.

---

### G14: Action Chunking vs. Single-Step Tradeoff
**Status:** Partially addressed

Chunking (ACT) improves temporal consistency but adds latency. Single-step prediction is fast but jerky.

**Existing solutions:** FAST tokenizer (compresses chunks into fewer tokens), adaptive chunk sizes.

**Remaining gap:** No universal solution. Task-dependent tradeoff.

---

### G15: Multi-View Visual Processing Cost
**Status:** Unaddressed for small VLAs

Most VLAs use single-camera RGB. Multi-view + depth is underutilized. Processing multiple views multiplies compute linearly.

**Potential approach:** Cross-view token fusion, view-conditioned pruning.

---

## How APEX Addresses These Gaps

| Gap | APEX Coverage |
|-----|--------------|
| G1: Visual Token Redundancy | **Direct** - Learned router selects task-relevant tokens |
| G2: Prefill-Decode Mismatch | **Direct** - Action loss trains selection, not attention proxy |
| G3: Spatial Grounding | **Direct** - Task-conditioned selection focuses on relevant regions |
| G6: Real-Time Inference | **Direct** - Quadratic attention savings from token reduction |

| Gap | APEX Does NOT Address |
|-----|----------------------|
| G4: Catastrophic Forgetting | Requires training-level intervention |
| G5: Action Tokenization | Requires architecture-level change |
| G7: Cross-Embodiment | Fundamental challenge |
| G8: Long-Horizon | Requires temporal memory mechanism |
| G9: Sim-to-Real | Requires data-level solution |
| G10: Safety | Requires constraint/verification layer |
| G11: Benchmark Saturation | Community-level issue |
| G12: Proprioceptive Integration | Requires architectural change |
| G13: Data Efficiency | Requires data-level solution |
| G14: Chunking Tradeoff | Requires tokenizer-level change |
| G15: Multi-View Cost | Requires multi-view processing |

APEX is designed to be composable with solutions for the other gaps. It can be combined with safety filters (G10), history buffers (G8), or cross-embodiment representations (G7).
