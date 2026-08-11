"""
tests/test_engine.py — Gradient verification for the Tensor autograd engine.

Every test compares analytical gradients (computed by `.backward()`)
against numerical gradients (computed via central difference,
h=1e-5), exactly mirroring ann-foundation's `test_engine.py` approach
of trusting nothing without numerical proof.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import Tensor, conv2d, max_pool2d


def numerical_grad(f, x_data, h=1e-5):
    """Central-difference numerical gradient of scalar function f w.r.t. x_data."""
    grad = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x_data[idx]
        x_data[idx] = orig + h
        f_plus = f()
        x_data[idx] = orig - h
        f_minus = f()
        x_data[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


# --------------------------------------------------------------------------
# Basic op gradients
# --------------------------------------------------------------------------

def test_add_gradient():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    c.backward()
    assert np.isclose(a.grad, 1.0)
    assert np.isclose(b.grad, 1.0)


def test_mul_gradient():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    c.backward()
    assert np.isclose(a.grad, 3.0)
    assert np.isclose(b.grad, 2.0)


def test_chain_rule():
    """c = a*b + a  =>  dc/da = b + 1, dc/db = a   (same example as ann-foundation README)"""
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b + a
    c.backward()
    assert np.isclose(a.grad, 4.0)
    assert np.isclose(b.grad, 2.0)


def test_pow_gradient():
    a = Tensor(3.0)
    b = a ** 2
    b.backward()
    assert np.isclose(a.grad, 6.0)  # d(a^2)/da = 2a = 6


def test_gradient_accumulation():
    """A Tensor used twice in the graph must SUM incoming gradients."""
    a = Tensor(3.0)
    b = a + a  # b = 2a, db/da should be 2, not 1 (would be a bug if grads overwrite)
    b.backward()
    assert np.isclose(a.grad, 2.0)


def test_relu_gradient():
    np.random.seed(0)
    x_data = np.random.randn(5, 5)

    def f():
        x = Tensor(x_data.copy())
        return x.relu().sum().data.item()

    x = Tensor(x_data.copy())
    out = x.relu().sum()
    out.backward()
    num = numerical_grad(f, x_data)
    assert np.allclose(x.grad, num, atol=1e-6)


def test_exp_log_gradient():
    np.random.seed(1)
    x_data = np.abs(np.random.randn(4)) + 0.5  # keep positive for log()

    def f():
        x = Tensor(x_data.copy())
        return (x.exp().log() * x).sum().data.item()

    x = Tensor(x_data.copy())
    out = (x.exp().log() * x).sum()
    out.backward()
    num = numerical_grad(f, x_data)
    assert np.allclose(x.grad, num, atol=1e-5)


def test_division_gradient():
    np.random.seed(2)
    x_data = np.random.randn(3, 3)
    y_data = np.random.randn(3, 3) + 3.0  # avoid near-zero denominators

    def f():
        x = Tensor(x_data.copy())
        y = Tensor(y_data.copy())
        return (x / y).sum().data.item()

    x = Tensor(x_data.copy())
    y = Tensor(y_data.copy())
    out = (x / y).sum()
    out.backward()
    num_x = numerical_grad(f, x_data)
    assert np.allclose(x.grad, num_x, atol=1e-5)

# --------------------------------------------------------------------------
# Broadcasting-aware backward — the key new mechanism vs. a scalar engine
# --------------------------------------------------------------------------

def test_broadcast_add_gradient_shapes():
    """Adding a (1, n) bias to a (batch, n) matrix must sum grad back to (1, n)."""
    x = Tensor(np.random.randn(4, 5))
    bias = Tensor(np.random.randn(1, 5))
    out = (x + bias).sum()
    out.backward()
    assert x.grad.shape == (4, 5)
    assert bias.grad.shape == (1, 5)

def test_broadcast_scalar_gradient():
    """Adding a scalar Tensor to a matrix should sum all contributions into the scalar's grad."""
    x = Tensor(np.random.randn(3, 3))
    scalar = Tensor(2.0)
    out = (x + scalar).sum()
    out.backward()
    assert np.isclose(scalar.grad, 9.0)  # scalar contributes to all 9 elements

def test_broadcast_add_gradient_values():
    np.random.seed(3)
    x_data = np.random.randn(4, 5)
    bias_data = np.random.randn(1, 5)

    def f():
        x = Tensor(x_data.copy())
        bias = Tensor(bias_data.copy())
        return (x + bias).sum().data.item()

    x = Tensor(x_data.copy())
    bias = Tensor(bias_data.copy())
    out = (x + bias).sum()
    out.backward()

    num_bias = numerical_grad(f, bias_data)
    assert np.allclose(bias.grad, num_bias, atol=1e-5)


# --------------------------------------------------------------------------
# Matmul
# --------------------------------------------------------------------------

def test_matmul_gradient_shapes():
    X = Tensor(np.random.randn(4, 3))
    W = Tensor(np.random.randn(3, 5))
    out = X.matmul(W).sum()
    out.backward()
    assert X.grad.shape == (4, 3)
    assert W.grad.shape == (3, 5)


def test_matmul_gradient_values():
    np.random.seed(4)
    X_data = np.random.randn(4, 3)
    W_data = np.random.randn(3, 5)

    def f():
        X = Tensor(X_data.copy())
        W = Tensor(W_data.copy())
        return X.matmul(W).sum().data.item()

    X = Tensor(X_data.copy())
    W = Tensor(W_data.copy())
    out = X.matmul(W).sum()
    out.backward()

    num_X = numerical_grad(f, X_data)
    num_W = numerical_grad(f, W_data)
    assert np.allclose(X.grad, num_X, atol=1e-5)
    assert np.allclose(W.grad, num_W, atol=1e-5)


# --------------------------------------------------------------------------
# Reshape / transpose
# --------------------------------------------------------------------------

def test_reshape_gradient():
    x = Tensor(np.random.randn(2, 6))
    out = x.reshape(3, 4).sum()
    out.backward()
    assert x.grad.shape == (2, 6)
    assert np.allclose(x.grad, np.ones((2, 6)))


def test_transpose_gradient():
    np.random.seed(5)
    x_data = np.random.randn(3, 4)

    def f():
        x = Tensor(x_data.copy())
        weights = Tensor(np.arange(12).reshape(4, 3).astype(np.float64))
        return (x.T * weights).sum().data.item()

    x = Tensor(x_data.copy())
    weights = Tensor(np.arange(12).reshape(4, 3).astype(np.float64))
    out = (x.T * weights).sum()
    out.backward()
    num = numerical_grad(f, x_data)
    assert np.allclose(x.grad, num, atol=1e-5)


# --------------------------------------------------------------------------
# Conv2D / MaxPool2D — the operations unique to a CNN, gradient-checked
# against numerical differentiation since these are the easiest ops to get
# subtly wrong (index bugs in padding/striding fail silently otherwise).
# --------------------------------------------------------------------------

def test_conv2d_gradient_shapes():
    x = Tensor(np.random.randn(2, 3, 8, 8))
    w = Tensor(np.random.randn(4, 3, 3, 3))
    b = Tensor(np.random.randn(4))
    out = conv2d(x, w, b, stride=1, pad=1)
    assert out.shape == (2, 4, 8, 8)
    out.sum().backward()
    assert x.grad.shape == x.shape
    assert w.grad.shape == w.shape
    assert b.grad.shape == b.shape


def test_conv2d_gradient_values_no_padding():
    np.random.seed(10)
    x_data = np.random.randn(2, 2, 5, 5) * 0.5
    w_data = np.random.randn(3, 2, 3, 3) * 0.5

    def f():
        x = Tensor(x_data.copy())
        w = Tensor(w_data.copy())
        return conv2d(x, w, stride=1, pad=0).sum().data.item()

    x = Tensor(x_data.copy())
    w = Tensor(w_data.copy())
    out = conv2d(x, w, stride=1, pad=0)
    out.sum().backward()

    num_x = numerical_grad(f, x_data)
    num_w = numerical_grad(f, w_data)
    assert np.allclose(x.grad, num_x, atol=1e-6)
    assert np.allclose(w.grad, num_w, atol=1e-6)


def test_conv2d_gradient_values_with_padding_and_stride():
    np.random.seed(11)
    x_data = np.random.randn(2, 2, 6, 6) * 0.5
    w_data = np.random.randn(3, 2, 3, 3) * 0.5
    b_data = np.random.randn(3) * 0.5

    def f():
        x = Tensor(x_data.copy())
        w = Tensor(w_data.copy())
        b = Tensor(b_data.copy())
        return conv2d(x, w, b, stride=2, pad=1).sum().data.item()

    x = Tensor(x_data.copy())
    w = Tensor(w_data.copy())
    b = Tensor(b_data.copy())
    out = conv2d(x, w, b, stride=2, pad=1)
    out.sum().backward()

    num_x = numerical_grad(f, x_data)
    num_w = numerical_grad(f, w_data)
    num_b = numerical_grad(f, b_data)
    assert np.allclose(x.grad, num_x, atol=1e-6)
    assert np.allclose(w.grad, num_w, atol=1e-6)
    assert np.allclose(b.grad, num_b, atol=1e-6)


def test_max_pool2d_gradient_shapes():
    x = Tensor(np.random.randn(2, 3, 8, 8))
    out = max_pool2d(x, pool_size=2, stride=2)
    assert out.shape == (2, 3, 4, 4)
    out.sum().backward()
    assert x.grad.shape == x.shape


def test_max_pool2d_gradient_values():
    np.random.seed(12)
    x_data = np.random.randn(2, 2, 6, 6)
    weights = np.random.randn(2, 2, 3, 3)

    def f():
        x = Tensor(x_data.copy())
        out = max_pool2d(x, pool_size=2, stride=2)
        return (out.data * weights).sum()

    x = Tensor(x_data.copy())
    out = max_pool2d(x, pool_size=2, stride=2)
    loss = (out * Tensor(weights, requires_grad=False)).sum()
    loss.backward()

    num_x = numerical_grad(f, x_data)
    assert np.allclose(x.grad, num_x, atol=1e-6)


def test_max_pool2d_routes_gradient_to_max_only():
    """Gradient should be zero everywhere except the argmax position in each window."""
    x_data = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # single 2x2 window, max is 4.0 at (1,1)
    x = Tensor(x_data)
    out = max_pool2d(x, pool_size=2, stride=2)
    out.sum().backward()
    expected = np.array([[[[0.0, 0.0], [0.0, 1.0]]]])
    assert np.allclose(x.grad, expected)


def test_conv_then_pool_end_to_end_gradient():
    """A conv2d -> relu -> max_pool2d pipeline should produce nonzero, finite gradients."""
    np.random.seed(13)
    x = Tensor(np.random.randn(2, 1, 8, 8) * 0.3)
    w = Tensor(np.random.randn(4, 1, 3, 3) * 0.3)
    b = Tensor(np.zeros(4))

    h = conv2d(x, w, b, stride=1, pad=1).relu()
    h = max_pool2d(h, pool_size=2, stride=2)
    loss = h.sum()
    loss.backward()

    assert np.all(np.isfinite(x.grad))
    assert np.all(np.isfinite(w.grad))
    assert np.any(w.grad != 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


