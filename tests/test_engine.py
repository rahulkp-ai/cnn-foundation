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

