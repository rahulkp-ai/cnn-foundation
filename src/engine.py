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


def _sum_to_shape(grad: np.ndarray, shape: tuple)  -> np.ndarray:
    """
    Reduce 'grad' down to 'shape' by summing over broadcasted dimensions.

    This is the inverse of NumPy broadcasting. If a Tensor of shape 'shape'
    was broadcast (during the forward pass) to participate in an op that
    produced a larger-shaped output, the gradient flowing back has the 
    *larger* shape. We must sum it back down to 'shape' before assigning it
    to that Tensor's '.grad', or shapes won't match.

    Two cases handled:
        1. Extra leading dimensions (e.g shape () broadcast to (3, 4)):
            sum over those leading axes entirely.
        2. Size-1 dimensions that were stretched (e.g shape (1, 4) broadcast
            to (3, 4)): sum over that axis, keeping it as size 1.
    """

    # Case 1: grad has more dimensions than the target shape - sum off the
    # extra leading axes (NumPy broadcasting aligns from thr right).
    # Remove extra leading dimensions

    ndims_added     = grad.ndim - len(shape)
    if ndims_added > 0:
        grad    = grad.sum(axis=tuple(range(ndims_added)))
    
    # Case 2: any dimension that was originally size-1 (and got stretched)
    # must be sumed back to size 1, keeping the dimension (keepdims).
    # Reduce broadcasted dimensions

    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad   = grad.sum(axis=axis, keepdims=True)
    
    return grad.reshape(shape)

class Tensor:
    """
    A multi-dimensional array with reverse-mode automatic differentiation.

    Parameters
    ----------
    data          : array-like
                    The underlying values. Converted to a NumPy float64 array.
    _childern     : tuple[Tensor,.....]
                    Internal use - the Tensors that produced this one (for graph building)
    _op           : str
                    Internal use - a label for the operation that produced this Tensor
                    (purely cosmetic / for debugging, mirros ann-foundation's Value).
    requires_grad : bool
                    If False, this Tensor never accumalates gradients and is treated as
                    a constant (useful for imput data /  labels that don't need grads).
    """ 

    __slots__   = ("data", "grad", "requires_grad","_backward","_prev","_op")

    def __init__(self, data, _childern=(), _op="", requires_grad=True):
        self.data           =   np.asarray(data, dtype=np.float64)
        self.grad           =   np.zeros_like(self.data)
        self.requires_grad  =   requires_grad
        self._backward      =   lambda: None
        self._prev          =   set(_childern)
        self._op            =   _op

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
        """
        Reset the gradient buffer to zero (call before each backward pass).
        """
        self.grad   = np.zeros_like(self.data)

    def item(self):
        """Return a Python scalar - only valid for size-1 Tensors."""
        return float(self.data.reshape(-1)[0])
    
    # ------------------------------------------------------------------
    # Core ops. Each op:
    #   1. Computes the forward value with NumPy.
    #   2. Wraps it in a new Tensor that records its parents (`_children`).
    #   3. Attaches a `_backward` closure implementing the LOCAL derivative,
    #      which is later called by `.backward()` in topological order.
    # ------------------------------------------------------------------

    def __add__(self, other):
        other   = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out     = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad  += _sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(out.grad, other.data.shape)
        
        out._backward      = _backward
        return out
    
    def __mul__(self, other):
        other   = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out     = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad   += _sum_to_shape(out.grad * other.grad, self.data.shape)
            if other.requires_grad:
                other.grad  += _sum_to_shape(out.grad * self.data, other.data.shape)
        out._backward   = _backward
        return out
    
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only scalar exponets supported"
        out     = Tensor(self.data ** exponent,(self,), f"**{exponent}")

        def _backward():
            if self.requires_grad:
                self.grad   += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward       = _backward
        return out
    
    def matmul(self, other):
        """
        Matrix multiplication. Supports the standard 2S case (batch, in) @
        (in, out) -> (batch, out), which is all 'Linear' layers need.

        Backward rule (standard matmul gradient):
            dL/dA   = dL/dC @ B.T
            dL/dB   = A.T @ dL/dC
        """
        
        assert isinstance(other, Tensor), "matmul requires a Tensor operand"
        out     = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self.grad   += out.grad @ other.data.T
            if other.requires_grad:
                other.grad  += self.data.T @ out.grad
        out._backward   = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out     = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            if self.requires_grad:
                grad    = out.grad
                if not keepdims and axis is not None:
                    grad    = np.expand_dims(grad, axis=axis)
                self.grad   += np.ones_lik(self.data) * grad
        out._backward       = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out     = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")

        if axis is None:
            n   = self.data.size
        else:
            n   = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a] for a in axis])
        
        def _backward():
            if self.requires_grad:
                grad    = out.grad
                if not keepdims and axis is not None:
                    grad    = np.expand_dims(grad, axis=axis)
                self.grad   += (np.ones_like(self.data) / n) * grad
        out._backward       = _backward
        return out

    def reshape(self, *shape):
        if len(shape)   == 1 and isinstance(shape[0], tuple):
            shape       = shape[0]
        out             = Tensor(self.data.reshape(shape), (self,), "reshape")
        
        def _backward():
            if self.requires_grad:
                self.grad   += out.grad.reshape(self.data.shape)
        out._backward       = _backward
        return out
    
    def transpose(self, *axes):
        axes    = axes if axes else None
        out     = Tensor(self.data.transpose(axes), (self,), "transpose")
    
        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad   +=  out.grad.transpose()
                else:
                    inv_axes    = np.argsort(axes)
                    self.grad   += out.grad.transpose(inv_axes)
        out._backward       = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def relu(self):
        out     = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad   +=  (self.data > 0).astype(np.float64)  * out.grad 
        out._backward   =   _backward
        return out

    def exp(self):
        out     = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad   += out.data * out.grad
        out._backward   = _backward
        return out

    def log(self):
        out     = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad   += (1.0 / elf.data) * out.grad
        out._backward   = _backward
        return out

    def getitem(self, idx):
        """
        Indexing / sclicing, exposed as a graph op so gradiens routes corretly
        """
        out     = Tensor(self.data[idx], (self), "getitem")

        def _backward():
            if self.requires_grad:
                full_grad   = np.zeros_like(self.data)
                np.add.at(full_grad, idx, out.grad)
                self.grad   += full_grad 
        out._backward   = _backward
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
        return self  + (-other if isinstance(other, Tensor) else -1.0 * other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rmul__(self, other):
        return self  * other
    
    def __truediv__(self, other):
        other   = other if isinstance(other, Tensor) else Tensor (other, requires_grad=False)
        return self * other ** -1.0
    
    def __rtruediv__(self, other):
        return other * self ** -1.0
    # ------------------------------------------------------------------
    # backward(): build topological order, then apply chain rule in reverse
    # ------------------------------------------------------------------