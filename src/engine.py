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

from __future__ import annotations
import numpy as np


def _sum_to_shape(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Reduce `grad` down to `shape` by summing over broadcasted dimensions.

    This is the inverse of NumPy broadcasting. If a Tensor of shape `shape`
    was broadcast (during the forward pass) to participate in an op that
    produced a larger-shaped output, the gradient flowing back has the
    *larger* shape. We must sum it back down to `shape` before assigning it
    to that Tensor's `.grad`, or shapes won't match.

    Two cases handled:
      1. Extra leading dimensions (e.g. shape () broadcast to (3, 4)):
         sum over those leading axes entirely.
      2. Size-1 dimensions that were stretched (e.g. shape (1, 4) broadcast
         to (3, 4)): sum over that axis, keeping it as size 1.
    """
    # Case 1: grad has more dimensions than the target shape — sum off the
    # extra leading axes (NumPy broadcasting always aligns from the right).
    ndims_added = grad.ndim - len(shape)
    if ndims_added > 0:
        grad = grad.sum(axis=tuple(range(ndims_added)))

    # Case 2: any dimension that was originally size-1 (and got stretched)
    # must be summed back to size 1, keeping the dimension (keepdims).
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    """
    A multi-dimensional array with reverse-mode automatic differentiation.

    Parameters
    ----------
    data : array-like
        The underlying values. Converted to a NumPy float64 array.
    _children : tuple[Tensor, ...]
        Internal use — the Tensors that produced this one (for graph building).
    _op : str
        Internal use — a label for the operation that produced this Tensor
        (purely cosmetic / for debugging, mirrors ann-foundation's Value).
    requires_grad : bool
        If False, this Tensor never accumulates gradients and is treated as
        a constant (useful for input data / labels that don't need grads).
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def zero_grad(self):
        """Reset the gradient buffer to zero (call before each backward pass)."""
        self.grad = np.zeros_like(self.data)

    def item(self):
        """Return a Python scalar — only valid for size-1 Tensors."""
        return float(self.data.reshape(-1)[0])

    # ------------------------------------------------------------------
    # Core ops. Each op:
    #   1. Computes the forward value with NumPy.
    #   2. Wraps it in a new Tensor that records its parents (`_children`).
    #   3. Attaches a `_backward` closure implementing the LOCAL derivative,
    #      which is later called by `.backward()` in topological order.
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only scalar exponents supported"
        out = Tensor(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            if self.requires_grad:
                self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def matmul(self, other):
        """
        Matrix multiplication. Supports the standard 2D case (batch, in) @
        (in, out) -> (batch, out), which is all `Linear` layers need.

        Backward rule (standard matmul gradient):
            dL/dA = dL/dC @ B.T
            dL/dB = A.T @ dL/dC
        """
        assert isinstance(other, Tensor), "matmul requires a Tensor operand"
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a] for a in axis])

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += (np.ones_like(self.data) / n) * grad
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else None
        out = Tensor(self.data.transpose(axes), (self,), "transpose")

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += out.grad.transpose()
                else:
                    inv_axes = np.argsort(axes)
                    self.grad += out.grad.transpose(inv_axes)
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def getitem(self, idx):
        """Indexing/slicing, exposed as a graph op so gradients route correctly."""
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward():
            if self.requires_grad:
                full_grad = np.zeros_like(self.data)
                np.add.at(full_grad, idx, out.grad)
                self.grad += full_grad
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        return self.getitem(idx)

    # ------------------------------------------------------------------
    # Reflected / derived arithmetic (mirrors ann-foundation's Value)
    # ------------------------------------------------------------------
    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -1.0 * other)

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * other ** -1.0

    def __rtruediv__(self, other):
        return other * self ** -1.0

    # ------------------------------------------------------------------
    # backward(): build topological order, then apply chain rule in reverse
    # ------------------------------------------------------------------
    def backward(self):
        """
        Run reverse-mode autodiff from this Tensor back through the graph.

        Identical algorithm to ann-foundation's `Value.backward()`: a DFS
        builds a topological ordering of the graph, then we seed this
        Tensor's gradient with ones (dL/dL = 1) and walk the order in
        reverse, calling each node's local `_backward`.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op='{self._op}')"


# ==========================================================================
# im2col / col2im — the trick that makes from-scratch convolution fast.
#
# A naive convolution implementation loops over every output pixel and every
# kernel position in pure Python, which is far too slow to train a real CNN.
# The standard trick (used internally by real frameworks too) is to unroll
# every receptive-field patch of the input into a column of a big matrix,
# so that convolution becomes a single matrix multiply:
#
#     conv(input, kernel)  ==  kernel_matrix @ im2col(input)
#
# These two functions are pure NumPy (no graph tracking) — they are used
# *inside* the conv2d Tensor op below, which handles the autograd bookkeeping
# around them. col2im is exactly the backward pass of im2col (scatter-add
# instead of gather), which is what makes conv2d's backward correct.
# ==========================================================================

def _im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """
    x: (N, C, H, W) -> columns: (C*kh*kw, N*out_h*out_w)

    Each column holds one flattened receptive-field patch, so a convolution
    becomes `W_flat @ columns`.
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    cols = np.zeros((C, kh, kw, N, out_h, out_w), dtype=x.dtype)
    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            cols[:, y, xk, :, :, :] = x_padded[:, :, y:y_max:stride, xk:x_max:stride].transpose(1, 0, 2, 3)

    cols = cols.reshape(C * kh * kw, N * out_h * out_w)
    return cols


def _col2im(cols: np.ndarray, x_shape: tuple, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """
    Inverse of `_im2col`: scatter-add columns back into an (N, C, H, W) image.

    This is exactly the backward pass of im2col — wherever im2col *read* a
    value into multiple columns (because overlapping receptive fields share
    pixels), col2im must *add* the incoming gradient back into every one of
    those positions, since the chain rule sums contributions from every path
    a value took through the forward computation.
    """
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    cols_reshaped = cols.reshape(C, kh, kw, N, out_h, out_w)
    x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)

    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            x_padded[:, :, y:y_max:stride, xk:x_max:stride] += cols_reshaped[:, y, xk, :, :, :].transpose(1, 0, 2, 3)

    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]


def conv2d(x: "Tensor", weight: "Tensor", bias: "Tensor" = None, stride: int = 1, pad: int = 0) -> "Tensor":
    """
    2D convolution as a Tensor graph op.

    Shapes
    ------
    x      : (N, C_in, H, W)
    weight : (C_out, C_in, kh, kw)
    bias   : (C_out,) or None
    output : (N, C_out, out_h, out_w)

    Implementation: im2col turns the convolution into one matmul
    (`weight_flat @ columns`), which is both fast and makes the backward
    pass simple — it's just a matmul backward, plus col2im to undo the
    patch-unrolling on the way back to the input's gradient.
    """
    N, C_in, H, W = x.data.shape
    C_out, _, kh, kw = weight.data.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    cols = _im2col(x.data, kh, kw, stride, pad)            # (C_in*kh*kw, N*out_h*out_w)
    w_flat = weight.data.reshape(C_out, -1)                  # (C_out, C_in*kh*kw)

    out_data = w_flat @ cols                                 # (C_out, N*out_h*out_w)
    out_data = out_data.reshape(C_out, N, out_h, out_w).transpose(1, 0, 2, 3)  # (N, C_out, out_h, out_w)

    if bias is not None:
        out_data = out_data + bias.data.reshape(1, C_out, 1, 1)

    parents = (x, weight) if bias is None else (x, weight, bias)
    out = Tensor(out_data, parents, "conv2d")

    def _backward():
        # Reshape incoming grad to match the matmul-form output: (C_out, N*out_h*out_w)
        grad_out = out.grad.transpose(1, 0, 2, 3).reshape(C_out, -1)

        if weight.requires_grad:
            # dL/dW_flat = grad_out @ cols.T  -> reshape back to (C_out, C_in, kh, kw)
            dW = grad_out @ cols.T
            weight.grad += dW.reshape(weight.data.shape)

        if x.requires_grad:
            # dL/d(cols) = W_flat.T @ grad_out, then scatter back to image via col2im
            d_cols = w_flat.T @ grad_out
            x.grad += _col2im(d_cols, x.data.shape, kh, kw, stride, pad)

        if bias is not None and bias.requires_grad:
            bias.grad += out.grad.sum(axis=(0, 2, 3))

    out._backward = _backward
    return out


def max_pool2d(x: "Tensor", pool_size: int = 2, stride: int = None) -> "Tensor":
    """
    2D max pooling as a Tensor graph op.

    Shapes: x (N, C, H, W) -> out (N, C, out_h, out_w)

    Backward rule: gradient flows ONLY to the position that was the max in
    each pooling window (every other position in the window gets zero
    gradient, since infinitesimally changing a non-max value doesn't change
    the max). We record the argmax positions on the forward pass and scatter
    the incoming gradient back to exactly those positions on the backward pass.
    """
    if stride is None:
        stride = pool_size
    N, C, H, W = x.data.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    out_data = np.zeros((N, C, out_h, out_w), dtype=x.data.dtype)
    # argmax_idx[n, c, i, j] stores the flat (row, col) offset within the
    # pooling window that achieved the max, for routing gradient on backward.
    argmax_row = np.zeros((N, C, out_h, out_w), dtype=np.int64)
    argmax_col = np.zeros((N, C, out_h, out_w), dtype=np.int64)

    for i in range(out_h):
        r0 = i * stride
        for j in range(out_w):
            c0 = j * stride
            window = x.data[:, :, r0:r0 + pool_size, c0:c0 + pool_size]  # (N, C, pool, pool)
            flat = window.reshape(N, C, -1)
            idx = np.argmax(flat, axis=-1)                                # (N, C)
            out_data[:, :, i, j] = np.take_along_axis(flat, idx[..., None], axis=-1).squeeze(-1)
            argmax_row[:, :, i, j] = idx // pool_size
            argmax_col[:, :, i, j] = idx % pool_size

    out = Tensor(out_data, (x,), "max_pool2d")

    def _backward():
        if x.requires_grad:
            dx = np.zeros_like(x.data)
            for i in range(out_h):
                r0 = i * stride
                for j in range(out_w):
                    c0 = j * stride
                    rows = r0 + argmax_row[:, :, i, j]   # (N, C)
                    cols_ = c0 + argmax_col[:, :, i, j]  # (N, C)
                    g = out.grad[:, :, i, j]              # (N, C)
                    for n in range(N):
                        for c in range(C):
                            dx[n, c, rows[n, c], cols_[n, c]] += g[n, c]
            x.grad += dx

    out._backward = _backward
    return out
