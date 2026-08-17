"""
model.py — CNN class: composes layers Sequential-style.

Mirrors how ann-foundation's `MLP` composed `Layer` objects (which
themselves composed `Neuron` objects). Here, a `CNN` composes arbitrary
`Layer` objects (Conv2D, MaxPool2D, Flatten, Linear) plus activation
functions, in the order given, and exposes the same `parameters()` /
`zero_grad()` interface the optimizer expects.

Activations (relu) are plain functions, not Layer objects with state, so
`Sequential` accepts a mix of Layer instances and callables — anything
that implements `__call__(x) -> Tensor` works as a pipeline stage.
"""

from __future__ import annotations
from .layers import Layer


class Sequential(Layer):
    """
    Chains a list of callables (Layer instances and/or plain functions like
    `relu`) into a single forward pass, applied in order.
    """

    def __init__(self, *stages):
        self.stages = list(stages)

    def __call__(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def parameters(self):
        params = []
        for stage in self.stages:
            if hasattr(stage, "parameters"):
                params.extend(stage.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self):
        lines = "\n  ".join(repr(s) if hasattr(s, "__repr__") else str(s) for s in self.stages)
        return f"Sequential(\n  {lines}\n)"


class CNN(Sequential):
    """
    Convenience subclass — purely cosmetic, gives the model a CNN-specific
    name in code/notebooks while reusing all of Sequential's composition
    logic. Construct it exactly like Sequential:

        model = CNN(
            Conv2D(1, 8, kernel_size=3, pad=1), relu,
            MaxPool2D(2),
            Conv2D(8, 16, kernel_size=3, pad=1), relu,
            MaxPool2D(2),
            Flatten(),
            Linear(16 * 7 * 7, 10),
        )
    """
    pass
