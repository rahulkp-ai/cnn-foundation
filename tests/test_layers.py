"""
tests/test_layers.py — Tests for stateful layer classes.

Covers shape correctness, parameter registration, zero_grad behavior, and
that gradients actually flow back to layer parameters through a forward +
backward pass (catching wiring bugs that pure functional tests in
test_engine.py wouldn't catch, since those don't involve a Layer's own
parameter storage).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import Tensor
from src.layers import Conv2D, MaxPool2D, Linear, Flatten


def test_conv2d_output_shape():
    layer = Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, pad=1)
    x = Tensor(np.random.randn(5, 3, 16, 16))
    out = layer(x)
    assert out.shape == (5, 8, 16, 16)  # pad=1, stride=1, k=3 preserves spatial size


def test_conv2d_output_shape_with_stride():
    layer = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=2, pad=1)
    x = Tensor(np.random.randn(2, 1, 9, 9))
    out = layer(x)
    expected_size = (9 + 2 * 1 - 3) // 2 + 1
    assert out.shape == (2, 4, expected_size, expected_size)


def test_conv2d_no_bias():
    layer = Conv2D(in_channels=2, out_channels=4, kernel_size=3, pad=1, bias=False)
    assert layer.bias is None
    assert len(layer.parameters()) == 1  # only weight, no bias


def test_conv2d_parameters_registered():
    layer = Conv2D(in_channels=2, out_channels=4, kernel_size=3, pad=1)
    params = layer.parameters()
    assert len(params) == 2  # weight + bias
    assert params[0].shape == (4, 2, 3, 3)
    assert params[1].shape == (4,)


def test_conv2d_xavier_init_not_degenerate():
    """Weights shouldn't all be zero or identical (a common init bug)."""
    layer = Conv2D(in_channels=3, out_channels=8, kernel_size=3)
    assert not np.allclose(layer.weight.data, 0.0)
    assert layer.weight.data.std() > 0


def test_conv2d_gradient_flows_to_parameters():
    layer = Conv2D(in_channels=1, out_channels=2, kernel_size=3, pad=1)
    x = Tensor(np.random.randn(2, 1, 6, 6))
    out = layer(x)
    loss = out.sum()
    layer.zero_grad()
    loss.backward()
    assert np.any(layer.weight.grad != 0)
    assert np.any(layer.bias.grad != 0)


def test_maxpool2d_output_shape():
    layer = MaxPool2D(pool_size=2, stride=2)
    x = Tensor(np.random.randn(3, 4, 8, 8))
    out = layer(x)
    assert out.shape == (3, 4, 4, 4)


def test_maxpool2d_has_no_parameters():
    layer = MaxPool2D(pool_size=2)
    assert layer.parameters() == []


def test_maxpool2d_default_stride_equals_pool_size():
    layer = MaxPool2D(pool_size=3)
    assert layer.stride == 3


def test_linear_output_shape():
    layer = Linear(in_features=10, out_features=4)
    x = Tensor(np.random.randn(7, 10))
    out = layer(x)
    assert out.shape == (7, 4)


def test_linear_no_bias():
    layer = Linear(in_features=5, out_features=3, bias=False)
    assert layer.bias is None
    assert len(layer.parameters()) == 1


def test_linear_parameters_registered():
    layer = Linear(in_features=6, out_features=2)
    params = layer.parameters()
    assert len(params) == 2
    assert params[0].shape == (6, 2)
    assert params[1].shape == (2,)


def test_linear_gradient_flows_to_parameters():
    layer = Linear(in_features=4, out_features=3)
    x = Tensor(np.random.randn(5, 4))
    out = layer(x)
    loss = out.sum()
    layer.zero_grad()
    loss.backward()
    assert np.any(layer.weight.grad != 0)
    assert np.any(layer.bias.grad != 0)


def test_flatten_output_shape():
    layer = Flatten()
    x = Tensor(np.random.randn(4, 3, 5, 5))
    out = layer(x)
    assert out.shape == (4, 3 * 5 * 5)


def test_flatten_preserves_values():
    layer = Flatten()
    x_data = np.random.randn(2, 2, 2, 2)
    x = Tensor(x_data)
    out = layer(x)
    assert np.allclose(out.data, x_data.reshape(2, -1))


def test_flatten_has_no_parameters():
    layer = Flatten()
    assert layer.parameters() == []


def test_flatten_gradient_passthrough():
    layer = Flatten()
    x = Tensor(np.random.randn(3, 2, 4, 4))
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad, np.ones_like(x.data))


def test_zero_grad_resets_gradients():
    layer = Linear(in_features=3, out_features=2)
    x = Tensor(np.random.randn(4, 3))
    out = layer(x)
    out.sum().backward()
    assert np.any(layer.weight.grad != 0)
    layer.zero_grad()
    assert np.all(layer.weight.grad == 0)
    assert np.all(layer.bias.grad == 0)


def test_conv_pool_linear_pipeline_shapes():
    """A realistic small pipeline should produce correctly-shaped output end to end."""
    conv = Conv2D(1, 4, kernel_size=3, pad=1)
    pool = MaxPool2D(2)
    flat = Flatten()
    fc = Linear(4 * 4 * 4, 10)

    x = Tensor(np.random.randn(6, 1, 8, 8))
    h = conv(x).relu()
    h = pool(h)
    h = flat(h)
    logits = fc(h)
    assert logits.shape == (6, 10)
