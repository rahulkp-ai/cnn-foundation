"""
activations.py — Activation functions, extending the core engine ops.

ReLU already exists as a method on Tensor (engine.relu()) since it's used
heavily enough to warrant being a core op. Softmax is defined here as a
standalone function built out of existing Tensor ops (exp, sum, div),
which means its backward pass is handled automatically by the graph —
we don't need to derive or implement a softmax-specific gradient by hand.

This mirrors how ann-foundation exposed tanh/relu/sigmoid directly on the
Value class — the same activations are useful building blocks, just now
operating on batched array data instead of scalars.
"""

from __future__ import annotations
import numpy as np
from .engine import Tensor


def relu(x: Tensor) -> Tensor:
    """Rectified Linear Unit: max(0, x). Thin wrapper around Tensor.relu()."""
    return x.relu()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax along `axis`, with the standard max-subtraction trick for
    numerical stability (prevents exp() overflow for large logits).

    Built entirely from existing differentiable Tensor ops (no new
    `_backward` needed here) — subtracting the max is implemented as a
    plain NumPy op on a *detached* constant, which is safe because
    softmax(x) == softmax(x - c) for any constant c, so subtracting the
    max doesn't change the true gradient, only improves numerics.
    """
    shifted_data = x.data - np.max(x.data, axis=axis, keepdims=True)
    shifted = Tensor(shifted_data, (x,), "softmax_shift")

    def _shift_backward():
        if x.requires_grad:
            x.grad += shifted.grad
    shifted._backward = _shift_backward

    exps = shifted.exp()
    return exps / exps.sum(axis=axis, keepdims=True)


def sigmoid(x: Tensor) -> Tensor:
    """Logistic sigmoid: 1 / (1 + e^-x). Included for completeness / reuse."""
    return Tensor(1.0, requires_grad=False) / (Tensor(1.0, requires_grad=False) + (-x).exp())
