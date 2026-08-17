"""
tests/test_optim.py — Tests for optimizers: SGD, SGD+Momentum, Adam.

Verifies each optimizer's update rule in isolation (so a bug in one
optimizer's math is caught directly, without needing a full training run
to surface it) plus integration tests confirming each optimizer reduces
loss on a small classification task.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import Tensor
from src.layers import Linear
from src.losses import cross_entropy
from src.optim import SGD, SGDMomentum, Adam


def test_sgd_update_rule():
    p = Tensor(np.array([1.0, 2.0, 3.0]))
    p.grad = np.array([0.1, 0.2, 0.3])
    opt = SGD([p], lr=0.5)
    opt.step()
    expected = np.array([1.0, 2.0, 3.0]) - 0.5 * np.array([0.1, 0.2, 0.3])
    assert np.allclose(p.data, expected)


def test_sgd_zero_grad():
    p = Tensor(np.array([1.0, 2.0]))
    p.grad = np.array([5.0, 5.0])
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    assert np.allclose(p.grad, 0.0)


def test_sgd_momentum_first_step_equals_plain_sgd_scaled():
    """On the first step, velocity starts at 0, so v = -lr*grad, update = v."""
    p = Tensor(np.array([1.0, 1.0]))
    p.grad = np.array([1.0, 2.0])
    opt = SGDMomentum([p], lr=0.1, momentum=0.9)
    opt.step()
    expected = np.array([1.0, 1.0]) + (-0.1 * np.array([1.0, 2.0]))
    assert np.allclose(p.data, expected)


def test_sgd_momentum_accumulates_velocity():
    """Second step's effective update should be larger in magnitude than plain SGD due to momentum."""
    p_momentum = Tensor(np.array([0.0]))
    opt_m = SGDMomentum([p_momentum], lr=0.1, momentum=0.9)

    p_plain = Tensor(np.array([0.0]))
    opt_p = SGD([p_plain], lr=0.1)

    # Same constant gradient each step (simulating a consistent descent direction)
    for _ in range(2):
        p_momentum.grad = np.array([1.0])
        opt_m.step()
        p_plain.grad = np.array([1.0])
        opt_p.step()

    # Momentum should have moved further than plain SGD after 2 consistent steps
    assert abs(p_momentum.data[0]) > abs(p_plain.data[0])


def test_adam_bias_correction_first_step():
    """
    Verify Adam's update against the closed-form first-step result.
    At t=1: m_hat = grad, v_hat = grad^2, so update = lr * sign(grad) (since
    grad / (|grad| + eps) ≈ sign(grad) when eps is small).
    """
    p = Tensor(np.array([0.0]))
    p.grad = np.array([2.0])
    opt = Adam([p], lr=0.1, eps=1e-8)
    opt.step()
    # update should be very close to -lr * 1.0 (sign of positive grad), since
    # m_hat/sqrt(v_hat) = grad/|grad| = 1 for any nonzero grad after bias correction
    assert np.isclose(p.data[0], -0.1, atol=1e-3)


def test_adam_timestep_increments():
    p = Tensor(np.array([1.0]))
    opt = Adam([p], lr=0.01)
    assert opt._t == 0
    p.grad = np.array([1.0])
    opt.step()
    assert opt._t == 1
    p.grad = np.array([1.0])
    opt.step()
    assert opt._t == 2


def test_optimizer_parameters_list_matches_input():
    p1 = Tensor(np.array([1.0]))
    p2 = Tensor(np.array([2.0, 3.0]))
    opt = SGD([p1, p2], lr=0.1)
    assert len(opt.parameters) == 2


# --------------------------------------------------------------------------
# Integration: each optimizer should reduce loss over training steps on a
# small classification task built from real Layer + loss machinery.
# --------------------------------------------------------------------------

def _train_tiny_classifier(opt_cls, opt_kwargs, steps=60, seed=0):
    np.random.seed(seed)
    X = np.random.randn(24, 5)
    y = np.random.randint(0, 3, size=24)

    fc1 = Linear(5, 8)
    fc2 = Linear(8, 3)
    params = fc1.parameters() + fc2.parameters()
    opt = opt_cls(params, **opt_kwargs)

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        x = Tensor(X)
        h = fc1(x).relu()
        logits = fc2(h)
        loss = cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.data.item())
    return losses


def test_sgd_reduces_loss():
    losses = _train_tiny_classifier(SGD, dict(lr=0.1))
    assert losses[-1] < losses[0]


def test_sgd_momentum_reduces_loss():
    losses = _train_tiny_classifier(SGDMomentum, dict(lr=0.1, momentum=0.9))
    assert losses[-1] < losses[0]


def test_adam_reduces_loss():
    losses = _train_tiny_classifier(Adam, dict(lr=0.05))
    assert losses[-1] < losses[0]


def test_adam_converges_faster_than_plain_sgd():
    """A reasonable expectation on this toy task: Adam should reach a lower
    loss than plain SGD within the same small number of steps."""
    sgd_losses = _train_tiny_classifier(SGD, dict(lr=0.1), steps=30)
    adam_losses = _train_tiny_classifier(Adam, dict(lr=0.05), steps=30)
    assert adam_losses[-1] < sgd_losses[-1]
