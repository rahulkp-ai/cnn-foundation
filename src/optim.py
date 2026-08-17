"""
optim.py — Optimizers: SGD, SGD with Momentum, and Adam.

ann-foundation's training loop did the parameter update manually:
    p.data -= learning_rate * p.grad

Here we generalize that single line into proper optimizer classes, since a
real CNN benefits noticeably from momentum/adaptive learning rates — plain
SGD on a multi-layer conv net converges slowly and is more sensitive to
learning rate choice.

Each optimizer exposes the same two-method interface:
    - `.step()`       : apply one parameter update using current `.grad` values
    - `.zero_grad()`  : zero every tracked parameter's gradient

This mirrors the optimizer API used by mainstream frameworks (intentionally
— it's a good interface to know) while every update rule is implemented
by hand here, no framework involved.
"""

from __future__ import annotations
import numpy as np


class Optimizer:
    """Base class — shared parameter storage and zero_grad."""

    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Vanilla stochastic gradient descent:
        p.data <- p.data - lr * p.grad
    """

    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad


class SGDMomentum(Optimizer):
    """
    SGD with momentum (classical / "heavy ball" momentum):
        v <- momentum * v - lr * grad
        p.data <- p.data + v

    Momentum accumulates a velocity term across steps, which damps
    oscillation in narrow valleys of the loss surface and tends to converge
    noticeably faster than vanilla SGD for deeper networks like a CNN.
    """

    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, v in zip(self.parameters, self._velocity):
            v *= self.momentum
            v -= self.lr * p.grad
            p.data += v


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation), Kingma & Ba (2014).

    Maintains exponential moving averages of the gradient (first moment,
    `m`) and the squared gradient (second moment, `v`), with bias
    correction for their initialization at zero:

        m <- beta1 * m + (1 - beta1) * grad
        v <- beta2 * v + (1 - beta2) * grad^2
        m_hat <- m / (1 - beta1^t)
        v_hat <- v / (1 - beta2^t)
        p.data <- p.data - lr * m_hat / (sqrt(v_hat) + eps)

    Adam adapts the effective learning rate per-parameter, which generally
    makes it a robust default for training CNNs without much learning-rate
    tuning.
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]
        self._t = 0

    def step(self):
        self._t += 1
        for p, m, v in zip(self.parameters, self._m, self._v):
            m *= self.beta1
            m += (1 - self.beta1) * p.grad
            v *= self.beta2
            v += (1 - self.beta2) * (p.grad ** 2)

            m_hat = m / (1 - self.beta1 ** self._t)
            v_hat = v / (1 - self.beta2 ** self._t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
