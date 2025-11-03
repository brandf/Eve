# Miriam: Orthogonalized and Spectrally-Stable Successor to Eve

## Overview

**Miriam** extends the **Eve** architecture by incorporating ideas **analogous to Muon**’s improvements over Adam.  
While **Eve** brought *Adam’s adaptive momentum and scaling concepts* into **inference dynamics**, **Miriam** brings *Muon’s orthogonalization and spectral stabilization* into the same framework.

> **Tagline:** *Miriam is to Eve what Muon is to Adam.*

Like her biblical counterpart to Moses, **Miriam** harmonizes and balances the system — bringing coherence and structure to Eve’s adaptive flow through orthogonalized residual updates and spectral control across depth.

---

## Concept Summary

### From Eve to Miriam

| Analogy | Training Axis | Inference Axis |
|----------|----------------|----------------|
| Base Optimizer | Adam | Eve |
| Orthogonalized Successor | Muon | Miriam |

Miriam aims to make the **forward dynamics** of transformers better-conditioned, smoother, and more isotropic by addressing the same problems Muon solved in **weight space**, but now in **representation space**.

---

## Motivation

**Eve’s limitation:** Though Eve stabilizes hidden-state evolution with adaptive momentum and scaling, residual updates (`g_ℓ`) may still be **ill-conditioned**, dominated by a few strong channels or heads. This can cause:

- Layer-to-layer **oscillation** (“depth chatter”)
- **Over-amplification** of certain directions in latent space
- **Vanishing participation** of weaker channels
- **Spectral drift** in very deep stacks

**Muon’s insight:** Many of these issues in optimization stem from similar root causes—low-rank or skewed update geometry.  
By orthogonalizing the update matrix and controlling spectral norms, Muon restored balance and conditioning in weight space.

**Miriam’s insight:** Do the same—**orthogonalize and spectrally normalize the residuals** in Eve’s forward pass.

---

## Core Mechanism

### Standard Eve Update (for reference)

\[
\begin{aligned}
m_{\ell+1} &= \beta_1 m_\ell + (1-\beta_1) g_\ell \\
v_{\ell+1} &= \beta_2 v_\ell + (1-\beta_2) g_\ell^2 \\
x_{\ell+1} &= x_\ell + \eta \frac{m_{\ell+1}}{\sqrt{v_{\ell+1}} + \epsilon}
\end{aligned}
\]

### Miriam Update

Before integrating momentum, **orthogonalize the residual** using a lightweight **Newton–Schulz iteration** and apply a **spectral norm guardrail**:

\[
\begin{aligned}
\widehat{g}_\ell &= \text{Orth}(g_\ell) \\
m_{\ell+1} &= \beta_1 m_\ell + (1-\beta_1) \widehat{g}_\ell \\
v_{\ell+1} &= \beta_2 v_\ell + (1-\beta_2) \widehat{g}_\ell^2 \\
x_{\ell+1} &= x_\ell + \eta \frac{m_{\ell+1}}{\sqrt{v_{\ell+1}} + \epsilon}
\end{aligned}
\]

where:

- **Orth(·)** uses one or more **Newton–Schulz** iterations to orthogonalize each token/channel submatrix.  
- A **spectral clip** optionally bounds the effective gain:  
  \( \widehat{g}_\ell \leftarrow \widehat{g}_\ell / \max(1, \|\widehat{g}_\ell\|_2 / s_{max}) \).

---

## Implementation Outline

### Files

```
nanochat/
  miriam/
    __init__.py
    dynamics.py   # Orthogonalization + spectral norm utilities
scripts/
  base_train.py   # add --miriam flag (mutually exclusive with --eve)
  mid_train.py    # same flag handling
```

### Configuration

```bash
--miriam
--miriam_beta1 0.9
--miriam_beta2 0.999
--miriam_eta 1.0
--miriam_eps 1e-8
--miriam_ns_steps 2
--miriam_smax 5.0
--miriam_decay 0.001
```

### Integration in Model Forward

```python
if config.miriam:
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

for block in self.blocks:
    g = block(x)

    if config.miriam:
        # Orthogonalize the residuals (token × channel)
        g = newton_schulz_orthogonalize(g, steps=config.miriam_ns_steps)

        # Optional spectral clipping
        sigma = g.norm(dim=-1, keepdim=True)
        g = g / torch.clamp(sigma / config.miriam_smax, min=1.0)

        # Eve-style adaptive momentum + scaling
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        x = x + eta * m / (v.sqrt() + eps)
    else:
        x = x + g
```

---

## Expected Effects

| Issue | Eve’s Fix | Remaining Problem | Miriam’s Fix |
|-------|------------|-------------------|---------------|
| Oscillatory updates | Momentum smoothing | Channel imbalance | Orthogonalization equalizes channels |
| Exploding/vanishing features | RMS scaling | Directional skew | Spectral norm clipping |
| Poor deep-layer stability | Adaptive step size | Depth spectral drift | Spectral regularization |
| Unequal feature utilization | None | Dominance of high-gain channels | Isotropic update geometry |

---

## Training & Evaluation

While Eve and Miriam affect the *forward pass* (not optimization), they can be directly compared on:

1. **Validation perplexity** (language modeling tasks)
2. **Hidden state smoothness:** cosine similarity between layer deltas
3. **Spectral statistics:** singular value histograms of residuals
4. **Activation variance across channels**
5. **Training stability** (loss curves, gradient norms)

**Goal:** Achieve equal or better performance with improved conditioning and smoother representational dynamics.

---

## Future Directions

- **Learned Orthogonalization:** Let the model learn how much to orthogonalize each layer dynamically.  
- **Spectral Energy Budgeting:** Constrain total “energy” across depth to prevent drift.  
- **Blockwise Orthogonalization:** Apply Newton–Schulz per attention head or FF block separately.  
- **Miriam‑Lite:** Orthogonalization only (no spectral clipping).  
- **Miriam‑Full:** Adds adaptive damping and energy normalization.  

---

## Acronym (optional backronym)

> **MIRIAM** — *Matrix‑Invariant Residual Integration Across Momentum*

---

## Naming Justification

Following the lineage of *Adam → Eve* and *Adam → Muon*, **Miriam** is the natural successor to **Eve**, paralleling **Muon’s** role relative to **Adam**:

- **Biblically:** Miriam is the **prophetess and sister of Moses** — counterpart to Muon’s (Moses’s) role.  
- **Conceptually:** She brings *harmony, rhythm, and balance* to what Moses (Muon) structured — perfectly describing an orthogonalized, spectrally balanced forward dynamic.  
- **Phonetically:** “Miriam” mirrors “Muon” with the same ‘M’ lineage, keeping the family sound.

> *Miriam: Harmonizing Eve’s Flow through Orthogonal and Spectral Depth Dynamics.*

---

## TL;DR Summary

- **Eve** adapted Adam’s training principles (momentum + adaptive scaling) to inference.  
- **Miriam** adapts Muon’s advances (orthogonalization + spectral control) to inference.  
- Implemented as a light modification to `gpt.py` in nanochat, it stabilizes residual geometry, enforces isotropy, and improves depth conditioning.  

> **Miriam is to Eve what Muon is to Adam — the next evolution of adaptive forward dynamics.**
