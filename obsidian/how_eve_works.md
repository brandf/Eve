# How Eve Works

## Overview
Eve treats residual connections as an adaptive integration problem across transformer depth. Each layer contributes an update `g = block(x)`, but rather than adding it directly, Eve keeps running moment estimates per token/channel:

```
m_{l+1} = beta1 * m_l + (1 - beta1) * g
v_{l+1} = beta2 * v_l + (1 - beta2) * g^2
x_{l+1} = x_l + eta * m_{l+1} / (sqrt(v_{l+1}) + eps)
```

This mirrors Adam’s optimizer update, but applied to hidden-state evolution through depth instead of parameter updates through time.

- Momentum buffer `m` smooths layer-to-layer oscillations.
- Variance buffer `v` normalizes per-channel activation energy.
- Step size `eta` controls the forward integration step, like a depth-wise learning rate.

Buffers reset each forward pass, so the accumulation is confined within one inference trajectory.

## Implementation Plan
1. Add an `--eve` flag (and hyperparameters) to the configurator so any training script can toggle the Eve update.
2. Modify the model forward to initialize zero velocity/variance tensors when Eve is enabled and replace the residual addition with the adaptive update.
3. Keep the classical `x = x + g` path for baseline runs, so existing nanochat workflows remain unchanged.
4. Use Adam-inspired defaults (`beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `eta=1.0`), with room to tune later.
5. Optional: note the analogy to AdamW—if we ever want an EveW, we’d decouple any depth-wise regularization from the adaptive step.

## Why It Maps to Adam
- **Momentum:** Adam’s first moment -> smooth residual flow between layers.
- **Adaptive scaling:** Adam’s second moment -> stabilize per-feature magnitudes across depth.
- **Step size:** Adam’s learning rate -> integration step for the hidden state dynamics.

Viewed this way, Eve is “Adam for inference”: the same stabilization tricks, but applied to the forward pass, making deep transformers behave like well-conditioned dynamical systems.
