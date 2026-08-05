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


