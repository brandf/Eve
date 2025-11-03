# Eve: Transformer Forward Dynamics with Momentum and Adaptive Scaling

## Overview

**Eve** is an architectural concept that reimagines transformer layer updates as an **adaptive, momentum-driven integration process**—analogous to the Adam optimizer, but applied to **hidden-state evolution through depth** rather than to weight updates through time.

In a standard transformer, each layer performs an Euler-like integration step:

\[
x_{\ell+1} = x_\ell + f_\ell(x_\ell)
\]

where \(f_\ell\) is the combined output of attention and MLP sublayers. Eve generalizes this by introducing **momentum** and **adaptive per-feature scaling**, so hidden states evolve more smoothly and stably as they propagate through the network stack.

---

## Concept Summary

### 1. The Problem: Euler-Style Instability Through Depth
Current transformers compute each residual update \(f_\ell(x_\ell)\) independently. This is analogous to **plain gradient descent** in optimization — simple, but prone to:

- **Oscillation**: hidden features “chatter” between layers.
- **Poor conditioning**: some channels dominate, others stagnate.
- **Flat regions**: layers output near-zero deltas, stalling representation flow.

### 2. The Insight: Layer Depth ≈ Integration Time
If each transformer layer represents an integration step through a continuous dynamical system, then residual updates are akin to velocity changes — just like integration in physics or optimization.

Thus, we can borrow from the success of **Adam (Adaptive Moment Estimation)**, which solved similar pathologies in gradient descent by tracking both the *momentum* and *variance* of gradients.

### 3. The Idea: Apply Adam-Like Dynamics Through Depth
Eve maintains two running moment estimates across layers for each token and channel:

\[
\begin{aligned}
m_{\ell+1} &= \beta_1 m_\ell + (1-\beta_1) f_\ell(x_\ell) \\
v_{\ell+1} &= \beta_2 v_\ell + (1-\beta_2) f_\ell(x_\ell)^2 \\
x_{\ell+1} &= x_\ell + \eta \frac{m_{\ell+1}}{\sqrt{v_{\ell+1}} + \epsilon}
\end{aligned}
\]

- **Momentum (m)**: smooths residual updates across depth (first moment).
- **Adaptive scaling (v)**: stabilizes channel magnitudes (second moment).
- **Step size (η)**: scales the update magnitude, analogous to learning rate.

This turns the forward pass into an **Adam-like integration process**, bringing the same benefits to hidden-state dynamics that Adam brought to optimization.

---

## Expected Benefits

| Challenge (today) | Effect | Eve Mechanism |
|--------------------|---------|----------------|
| Rapid oscillation of features between layers | unstable activations, noisy convergence | Momentum term smooths updates |
| Channel-scale imbalance | exploding/vanishing residuals | Adaptive scaling normalizes step per feature |
| Flat or low-gradient layers | information stalls | Momentum carries signal through flat zones |
| Deep stacks hard to train | vanishing residual energy | Inertia preserves long-range propagation |

---

## Architectural Changes (Minimal)

### Base formulation (in `gpt.py` or equivalent)

Replace the standard residual update:
```python
x = x + g    # g = block(x)
```
with:
```python
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * (g * g)
x = x + eta * m / (v.sqrt() + eps)
```

#### Parameters
| Symbol | Meaning | Typical Value |
|---------|----------|----------------|
| β₁ | Momentum decay | 0.9 |
| β₂ | Variance decay | 0.999 |
| ε | Stability constant | 1e‑8 |
| η | Step size | 1.0 |

#### State
Each token maintains two buffers per layer stack:
- `m`: velocity (first moment)
- `v`: variance (second moment)

These are reset each forward pass (no recurrence between batches).

---

## Training Compatibility

Eve operates **only in the forward pass**. It does not alter optimization or backprop mechanics.

- No change to the optimizer (AdamW etc. still fine).
- No change to loss computation.
- Memory overhead: +2× `[B, T, C]` tensors per forward (≈10–20% VRAM increase).
- Compatible with FlashAttention and checkpointing.

---

## Evaluation Plan

### 1. Baseline
Train nanochat (or similar GPT) with standard residuals.
Measure:
- Validation perplexity
- Stability (activation variance per layer)
- Gradient norm consistency

### 2. Eve Variant
Enable the new dynamics (`--eve` flag). Keep everything else constant.
Log:
- Mean |m| and |v| across depth
- Cosine similarity between consecutive layer updates (expected ↑)
- Same metrics as baseline

### 3. Compare
| Metric | Expected Trend |
|---------|----------------|
| Validation perplexity | same or slightly better |
| Activation smoothness | higher stability |
| Gradient flow | more consistent through depth |
| Training loss | smoother convergence curve |

### 4. Qualitative Analysis
- Plot hidden-state trajectory norms across layers (should show smoother progression).
- Visualize cosine similarity between layer updates (should be less noisy).
- Observe sample generation quality and consistency.

---

## Implementation Outline

### Files
```
nanochat/
  eve/
    __init__.py
    dynamics.py     # implements the Eve forward logic
scripts/
  base_train.py     # add --eve flag
  mid_train.py      # add --eve flag
```

### Flag (CLI)
```bash
--eve               # enable adaptive forward dynamics
--eve_beta1 0.9
--eve_beta2 0.999
--eve_eta 1.0
--eve_eps 1e-8
```

### Integration
In `GPT.forward()`:
```python
m = v = None
if config.eve:
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

for block in self.blocks:
    g = block(x)
    if not config.eve:
        x = x + g
    else:
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        x = x + eta * m / (v.sqrt() + eps)
```

---

## Theoretical Framing

Eve can be viewed as introducing a **second‑order dynamical system** through layer depth:

\[
\frac{d^2x}{d\ell^2} + \gamma \frac{dx}{d\ell} = F(x)
\]

where \(F(x)\) represents attention/MLP “forces,” and the discrete update uses exponential moving averages to approximate a *damped symplectic integrator*. This bridges residual networks, momentum ODEs, and adaptive normalization — giving a unifying physics‑inspired view of transformers as continuous dynamical systems.

---

## Future Directions

- **Learned β₁/β₂ per layer** — dynamic adaptation of inertia.
- **Cross-token coupling** — shared adaptive statistics between tokens.
- **Recurrent memory variant** — retain velocity across time steps (temporal Eve).
- **Energy‑based training** — constrain total kinetic energy across depth.
- **Eve‑Lite** — single‑moment version (momentum only, no variance).

---

## Naming

**Eve** — the conceptual successor to Adam.

- Symbolizes an evolution of Adam’s principles from optimization to representation dynamics.  
- Elegant and simple — no suffix needed.  
- Evokes continuity, origin, and progression — apt for a model where information flows naturally through layers.

**Tagline:** 
> Eve is to inference what Adam is to training.
> *Eve: Adaptive Momentum Propagation Through Depth*

---

## TL;DR Summary

Eve extends the transformer’s residual update rule with **momentum and adaptive scaling across depth**, creating a forward‑integration process that mirrors the benefits of the Adam optimizer — but applied to **hidden states** instead of **weights**.  
This provides smoother, more stable evolution through layers, potentially improving deep‑stack efficiency, stability, and coherence of learned representations.
