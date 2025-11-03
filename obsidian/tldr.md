# Eve TL;DR

- Adam’s momentum term (first moment) -> Eve’s velocity buffer across depth; smooths residual updates layer-to-layer.
- Adam’s adaptive variance (second moment) -> Eve’s per-channel scaling; prevents a few features from dominating the residual stream.
- Adam’s step size (learning rate) -> Eve’s integration step across layers; sets how aggressively hidden states advance through the stack.

Result: the forward pass mimics Adam’s stabilization tricks, but the “time” axis is depth instead of optimization steps.
