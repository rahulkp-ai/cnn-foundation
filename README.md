# CNN Foundation

> A convolutional neural network — Tensor autograd engine, `im2col`-based convolution, pooling, and a full training loop — built from scratch in pure Python and NumPy. No PyTorch, no TensorFlow.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-52%20passed-1D9E75?style=flat-square)](tests/)
[![CNN Foundation CI Pipeline](https://github.com/rahulkp-ai/cnn-foundation/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulkp-ai/cnn-foundation/actions/workflows/ci.yml)

---

## Overview

This repository implements a **NumPy-backed automatic differentiation engine** (`Tensor`) and a **Convolutional Neural Network** built entirely on top of it — convolution, pooling, activations, loss, and optimizers, all with hand-derived and numerically-verified backward passes.

It is the direct sequel to [**ann-foundation**](https://github.com/rahulkp-ai/ann-foundation), which built a scalar autograd engine (`Value`) and a Multi-Layer Perceptron from scratch. `cnn-foundation` lifts the same philosophy — understand every gradient, trust nothing without numerical proof — one level up: from scalars to N-dimensional arrays, and from fully-connected layers to convolution.

The project covers the complete CNN pipeline:

```
Image Input → Conv2D → ReLU → MaxPool2D → Conv2D → ReLU → MaxPool2D → Flatten → Linear → Softmax/CrossEntropy
                                          ↓
                        Backward Pass (im2col / col2im, chain rule) → Optimizer Step → Repeat
```

---

## Features

- **Tensor Autograd Engine** — reverse-mode automatic differentiation over NumPy arrays via topological sort
- **Broadcasting-Aware Backward** — gradients correctly "un-broadcast" back to their original shape (e.g. shared bias terms)
- **`im2col` / `col2im` Convolution** — the standard trick that turns convolution into a single matmul, made fast *and* differentiable
- **Max Pooling** — gradient routes only to the argmax position in each window
- **Activations** — ReLU (core op), Softmax and Sigmoid (composed from existing differentiable ops)
- **Cross-Entropy Loss** — numerically stable log-softmax via the log-sum-exp trick, with a clean closed-form gradient
- **Three Optimizers** — SGD, SGD with Momentum, and Adam, all implemented from their update equations
- **Layer / Model Composition** — `Conv2D`, `MaxPool2D`, `Linear`, `Flatten` compose via a `Sequential`/`CNN` container, mirroring `Neuron → Layer → MLP` from ann-foundation
- **MNIST Training Pipeline** — minibatching, training-curve plots, sample-prediction grids, and a confusion matrix
- **52 Tests** — every gradient verified against numerical differentiation (central difference)

---

## Project Structure

```
cnn-foundation/
├── src/
│   ├── engine.py        # Tensor class — autograd engine, im2col/col2im, conv2d, max_pool2d
│   ├── layers.py         # Conv2D, MaxPool2D, Linear, Flatten (stateful, learnable layers)
│   ├── activations.py    # relu, softmax, sigmoid
│   ├── losses.py         # cross_entropy (stable log-softmax + NLL), accuracy
│   ├── optim.py           # SGD, SGDMomentum, Adam
│   ├── model.py            # Sequential / CNN — composes layers
│   └── utils.py             # MNIST loading (+ synthetic fallback), batching, plotting
├── notebooks/
│   └── Tensor-Autograd.ipynb              # Step-by-step engine walkthrough with gradient checks
├── examples/
│   └── MNIST-Training-Visualization.ipynb # Full training run + visualizations
├── tests/
│   ├── test_engine.py    # Gradient checks: add, mul, matmul, broadcasting, conv2d, max_pool2d
│   ├── test_layers.py    # Shape, parameter registration, gradient-flow checks for each layer
│   └── test_optim.py     # Update-rule correctness + loss-reduction checks for each optimizer
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/rahulkp-ai/cnn-foundation.git
cd cnn-foundation

# Install core package
pip install -e .

# Install with dev tools (notebooks + testing)
pip install -e ".[dev]"
```

---

## Quick Start

```python
import numpy as np
from src.engine import Tensor
from src.layers import Conv2D, MaxPool2D, Flatten, Linear
from src.activations import relu
from src.model import CNN
from src.losses import cross_entropy, accuracy
from src.optim import Adam

# --- Tensor autograd engine ---
a = Tensor(2.0)
b = Tensor(3.0)
c = a * b + a       # c = 8.0
c.backward()

print(a.grad)       # 4.0  (dc/da = b + 1 = 4)
print(b.grad)       # 2.0  (dc/db = a = 2)

# --- Build a CNN ---
model = CNN(
    Conv2D(in_channels=1, out_channels=8, kernel_size=3, pad=1), relu,
    MaxPool2D(pool_size=2),                                          # 28x28 -> 14x14
    Conv2D(in_channels=8, out_channels=16, kernel_size=3, pad=1), relu,
    MaxPool2D(pool_size=2),                                          # 14x14 -> 7x7
    Flatten(),
    Linear(in_features=16 * 7 * 7, out_features=10),
)

optimizer = Adam(model.parameters(), lr=0.001)

# --- Train on a batch of (synthetic, here) MNIST-shaped data ---
X = np.random.randn(32, 1, 28, 28)
y = np.random.randint(0, 10, size=32)

for epoch in range(10):
    optimizer.zero_grad()
    logits = model(Tensor(X))
    loss = cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.data.item():.4f} | Acc: {accuracy(logits, y):.3f}")
```

For a real training run on actual MNIST data with visualizations, see [`examples/MNIST-Training-Visualization.ipynb`](examples/MNIST-Training-Visualization.ipynb).

---

## The Autograd Engine: `Tensor`

`Tensor` extends ann-foundation's scalar `Value` design to N-dimensional NumPy arrays. The core mechanism is unchanged — every op records its parents and a local `_backward` closure, and `.backward()` walks a topological ordering of the graph in reverse, applying the chain rule. Two things are genuinely new at the array level:

| Mechanism | Why it's needed |
|---|---|
| **Broadcasting-aware backward** | NumPy silently broadcasts shapes (e.g. a `(1, n)` bias added to a `(batch, n)` matrix). The backward pass must sum the incoming gradient back down to the original shape, or parameter gradients won't match parameter shapes. |
| **`im2col` / `col2im`** | Convolution implemented as nested loops is too slow to train anything real. `im2col` unrolls receptive-field patches into columns so convolution becomes one matmul; `col2im` is its exact backward (scatter-add) counterpart. |

```python
x = Tensor(0.5)
print(x.relu())     # Tensor(shape=(), op='relu')
print(x.exp())       # Tensor(shape=(), op='exp')
```

---

## Convolution & Pooling

```python
from src.engine import conv2d, max_pool2d

x = Tensor(np.random.randn(2, 1, 28, 28))      # (N, C_in, H, W)
w = Tensor(np.random.randn(8, 1, 3, 3))         # (C_out, C_in, kh, kw)
b = Tensor(np.zeros(8))

out = conv2d(x, w, b, stride=1, pad=1)           # (2, 8, 28, 28) — padding preserves spatial size
out = max_pool2d(out, pool_size=2, stride=2)      # (2, 8, 14, 14)
```

Max pooling's backward pass routes gradient **only** to the position that achieved the max in each window — every other position receives zero gradient, since an infinitesimal change to a non-max value doesn't change the max.

---

## Running Tests

```bash
pytest tests/ -v
```

All 52 tests verify analytical gradients against numerical approximations (central difference, h=1e-5), check layer shape correctness and parameter registration, and confirm each optimizer actually reduces loss:

```
tests/test_engine.py::test_add_gradient                                PASSED
tests/test_engine.py::test_chain_rule                                   PASSED
tests/test_engine.py::test_broadcast_add_gradient_values                PASSED
tests/test_engine.py::test_conv2d_gradient_values_with_padding_and_stride PASSED
tests/test_engine.py::test_max_pool2d_routes_gradient_to_max_only       PASSED
tests/test_layers.py::test_conv2d_gradient_flows_to_parameters          PASSED
tests/test_optim.py::test_adam_converges_faster_than_plain_sgd          PASSED
... (52 total)
```

---

## Learning Path

This repository is part of a broader AI/ML learning journey:

1. Mathematics for Computing
2. Linear Algebra
3. Artificial Neural Networks — [ann-foundation](https://github.com/rahulkp-ai/ann-foundation)
4. **Convolutional Neural Networks** ← you are here
5. Generative AI Systems

---

## Author

**RAHUL K P**
MSc Computer Science — University of Calicut (2026)
[GitHub](https://github.com/rahulkp-ai) · [LinkedIn](https://www.linkedin.com/in/rahulkp-ai/) · [Kaggle](https://www.kaggle.com/rahulkpai)

---

## License

MIT — see [LICENSE](LICENSE)
