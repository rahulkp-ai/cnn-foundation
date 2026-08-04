"""
engine.py — Core Tensor class: a NumPy-backed reverse-mode autograd engine.

This is the cnn-foundation analogue of ann-foundation's `Value` class, lifted
from scalars to N-dimensional arrays. The design is intentionally similar:

    - Every Tensor remembers the operation that produced it (`_prev`, `_backward`)
    - `.backward()` builds a topological order of the computation graph and
      walks it in reverse, accumulating gradients via the chain rule.
    - Gradients accumulate (+=) rather than overwrite, so a Tensor used in
      multiple places in the graph correctly sums incoming gradients.

The one genuinely new piece of machinery vs. a scalar engine is BROADCASTING.
NumPy silently broadcasts shapes on the forward pass (e.g. adding a (1, n)
bias to a (batch, n) matrix). On the backward pass we must "undo" that
broadcast by summing the incoming gradient back down to the original
(pre-broadcast) shape — otherwise gradient shapes won't match parameter
shapes and parameter updates will fail or silently corrupt data.

No PyTorch, no TensorFlow, no autograd library — just NumPy and chain rule.
"""