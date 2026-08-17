"""
layers.py — Stateful layers: Conv2D, MaxPool2D, Linear, Flatten.

These wrap the functional ops in `engine.py` (conv2d, max_pool2d, matmul)
with learnable parameters, mirroring how ann-foundation's `Neuron`/`Layer`
wrapped scalar Value weights. Each layer exposes:

    - `__call__(x)`      : forward pass, returns a Tensor (graph-tracked)
    - `.parameters()`    : list of learnable Tensors (for the optimizer)
    - `.zero_grad()`     : reset all parameter gradients to zero

Weight initialization uses Xavier/Glorot-style scaling (drawing on the
lesson from ann-foundation's training-bug fix, where bad initialization
caused loss to plateau — getting init right matters even more here since
CNNs are deeper).
"""

from __future__ import annotations
import numpy as np
from .engine import Tensor, conv2d, max_pool2d


class Layer:
    """Base class — just defines the shared interface."""

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x):
        raise NotImplementedError


class Conv2D(Layer):
    """
    2D convolutional layer.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int (square kernels only, e.g. 3 -> 3x3)
    stride, pad : int
    bias : bool — whether to learn an additive bias per output channel

    Weight shape: (out_channels, in_channels, kernel_size, kernel_size)
    Initialized with Xavier/Glorot uniform scaling based on fan-in, which
    keeps activation variance stable across layers at the start of training
    — the same fix that resolved ann-foundation's stuck-loss bug.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0, bias=True):
        self.stride = stride
        self.pad = pad
        fan_in = in_channels * kernel_size * kernel_size
        limit = np.sqrt(6.0 / (fan_in + out_channels * kernel_size * kernel_size))
        w_data = np.random.uniform(-limit, limit, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.weight = Tensor(w_data)
        self.bias = Tensor(np.zeros(out_channels)) if bias else None

    def __call__(self, x):
        return conv2d(x, self.weight, self.bias, stride=self.stride, pad=self.pad)

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f"Conv2D(weight={self.weight.shape}, stride={self.stride}, pad={self.pad})"


class MaxPool2D(Layer):
    """2D max pooling layer. No learnable parameters."""

    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def __call__(self, x):
        return max_pool2d(x, pool_size=self.pool_size, stride=self.stride)

    def parameters(self):
        return []

    def __repr__(self):
        return f"MaxPool2D(pool_size={self.pool_size}, stride={self.stride})"


class Linear(Layer):
    """
    Fully connected layer: y = x @ W + b

    Parameters
    ----------
    in_features, out_features : int
    bias : bool

    Weight shape: (in_features, out_features). Xavier-uniform init, same
    rationale as Conv2D.
    """

    def __init__(self, in_features, out_features, bias=True):
        limit = np.sqrt(6.0 / (in_features + out_features))
        w_data = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.weight = Tensor(w_data)
        self.bias = Tensor(np.zeros(out_features)) if bias else None

    def __call__(self, x):
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f"Linear(in={self.weight.shape[0]}, out={self.weight.shape[1]})"


class Flatten(Layer):
    """
    Flattens all dimensions after the batch dimension: (N, C, H, W) -> (N, C*H*W).

    Needed to transition from convolutional feature maps to a fully
    connected classifier head.
    """

    def __call__(self, x):
        n = x.shape[0]
        flat_size = int(np.prod(x.shape[1:]))
        return x.reshape(n, flat_size)

    def parameters(self):
        return []

    def __repr__(self):
        return "Flatten()"
