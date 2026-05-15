# APEX Methodology

## Overview

APEX is a learned, task-conditioned visual token router that sits between the frozen vision encoder and the frozen VLA transformer. It predicts which visual tokens are relevant to the current task and selects only those for the VLA to process.

## Design Principles

1. **Minimal footprint**: ~33K parameters, 5-15 min training
2. **Plug-and-play**: Wraps any VLA without architectural modification
3. **Frozen base**: 100% of VLA weights remain unchanged
4. **Multi-objective**: Simultaneously improves speed, accuracy, and spatial grounding

## Architecture Details

### The Router

The APEX router is a 2-layer MLP:

```
Input: [v_n ; l_pool]  (2d dimensions)
  -> Linear(2d, 64) + ReLU + Dropout(0.1)
  -> Linear(64, 1) + Sigmoid
Output: w_n in (0, 1)
```

Total parameters: 2d * 64 + 64 + 64 + 1

For d=512: 65,601 params
For d=256: 32,897 params
For d=128: 16,449 params

### Task Vector Extraction

The task vector is computed by mean-pooling the language tokens:

```
l_pool = mean(language_tokens, dim=sequence)
```

This is parameter-free and captures the overall task semantics.

### Differentiable Selection

During training, we use Gumbel-Softmax to make discrete top-K selection differentiable:

1. Compute log-importance: `log_w = log(weights)`
2. Add Gumbel noise: `noisy = (log_w + gumbel_noise) / tau`
3. Softmax: `soft_weights = softmax(noisy)`
4. Hard top-K: `indices = topk(soft_weights, K)`

The gradients flow through `soft_weights` (straight-through estimator).

At inference, we use deterministic top-K (no noise, tau -> 0).

### Temperature Annealing

The Gumbel temperature tau controls exploration vs. exploitation:
- tau = 1.0: High exploration, soft selection
- tau = 0.1: Low exploration, near-deterministic selection

We use cosine annealing from 1.0 to 0.1 over the first 2000 training steps.

## Training

### Loss Function

```
L = L_action + 0.1 * L_coverage + 0.05 * L_diversity
```

**L_action**: MSE between VLA's predicted action and ground-truth action. This is the primary signal that teaches the router which tokens matter.

**L_coverage**: Encourages selected tokens to cover the task-relevant spatial region. Uses a Gaussian kernel centered on the target position.

**L_diversity**: Penalizes selecting only spatially adjacent tokens. Encourages spatial spread of selected tokens.

### Training Procedure

1. Load frozen VLA and its training demonstrations
2. For each epoch:
   a. Sample batch of (image, language, action) triples
   b. Extract visual and language tokens from frozen VLA
   c. Run APEX router to get importance weights
   d. Select top-K tokens via Gumbel-TopK
   e. Forward selected tokens through frozen VLA
   f. Compute loss and backprop through router only
3. Total training time: 5-15 minutes on a single GPU

### Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| target_ratio | 0.25 | 0.1-0.5 | Fraction of tokens to keep |
| hidden_dim | 64 | 32-128 | Router MLP hidden size |
| lr | 3e-4 | 1e-4 to 1e-3 | Learning rate |
| dropout | 0.1 | 0.0-0.3 | Router dropout |
| tau_start | 1.0 | 0.5-2.0 | Initial Gumbel temperature |
| tau_end | 0.1 | 0.01-0.5 | Final Gumbel temperature |
| lambda_coverage | 0.1 | 0.01-1.0 | Coverage loss weight |
| lambda_diversity | 0.05 | 0.01-0.5 | Diversity loss weight |

## Computational Analysis

### Inference Overhead

The APEX router adds:
- MLP forward: ~200K multiply-adds (for N=196, d=512, hidden=64)
- Top-K selection: O(N log K) comparisons
- Total: ~0.02ms on GPU

### VLA Savings

Without APEX: VLA processes N visual tokens
With APEX: VLA processes K visual tokens (K < N)

Attention cost scales as O((M + N)^2) where M = language tokens.

For M=40, N=196, K=50:
- Without: (40 + 196)^2 = 55,696 per layer
- With: (40 + 50)^2 = 8,100 per layer
- Speedup: 6.9x in attention

End-to-end speedup (including ViT and action head): ~1.5-2x

### Memory Savings

- Router: ~130KB (33K params * 4 bytes)
- Selected tokens: reduces VLA activation memory proportionally to token reduction
- Net memory: slight reduction (saves more than the router costs)
