import numpy as np
from autograd.conv_utils import unfold as _unfold, fold as _fold


def unbroadcast(grad, shape):
    """Sum grad back down to shape, undoing numpy broadcasting"""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    """A Tensor wrapping a numpy array and supporting automatic differentiation"""
    def __init__(self, data, inputs=()):
        self.data = np.asarray(data, dtype=float)   # put data in a numpy array
        self.grad = np.zeros_like(self.data)        # gradient of this tensor
        self.grad_fn = lambda: None                 # backward rule, sends the new gradient to the inputs
        self.inputs = set(inputs)                   # input tensors producing this tensor

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))

        def grad_fn():
            # d(a+b)/da = 1, so gradient passes through
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out.grad_fn = grad_fn
        
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other))

        def grad_fn():
            # d(a*b)/da = b , so the gradient is scaled by the other's value
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        out.grad_fn = grad_fn

        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def grad_fn():
            # d(A@B)/dA = B, transposed so the shapes line up
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out.grad_fn = grad_fn

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def grad_fn():
            # d(sum(a))/da = 1
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out.grad_fn = grad_fn

        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,))

        def grad_fn():
            # d(-a)/da = -1
            self.grad += -out.grad
        out.grad_fn = grad_fn

        return out

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only numeric powers"
        out = Tensor(self.data ** k, (self,))

        def grad_fn():
            # d(a**k)/da = k * a**(k-1)
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out.grad_fn = grad_fn

        return out

    def mean(self):
        return self.sum() * Tensor(1.0 / self.data.size)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))

        def grad_fn():
            # d(relu(a))/da = 1 if a > 0, else 0
            self.grad += (self.data > 0) * out.grad
        out.grad_fn = grad_fn

        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    def zero_grad(self):
        # Reset the gradient to zero (e.g. before a new backward pass)
        self.grad = np.zeros_like(self.data)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def grad_fn():
            # d(exp(a))/da = exp(a)
            self.grad += out.data * out.grad
        out.grad_fn = grad_fn

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))

        def grad_fn():
            # d(log(a))/da = 1/a
            self.grad += (1.0 / self.data) * out.grad
        out.grad_fn = grad_fn

        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))

        def grad_fn():
            # Reshaping doesn't change the gradient, only the shape changes
            self.grad += out.grad.reshape(self.data.shape)
        out.grad_fn = grad_fn

        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,))

        def grad_fn():
            # Transpose doesn't change the gradient, only the shape changes
            self.grad += out.grad.transpose(np.argsort(axes))
        out.grad_fn = grad_fn

        return out

    def unfold(self, kh, kw, stride=1, pad=0):
        out = Tensor(_unfold(self.data, kh, kw, stride, pad), (self,))

        def grad_fn():
            # fold is the backward pass of unfold
            self.grad += _fold(out.grad, self.data.shape, kh, kw, stride, pad)
        out.grad_fn = grad_fn

        return out

    def max(self, axis, keepdims=False):
        idx = self.data.argmax(axis=axis)
        picked = np.expand_dims(idx, axis)
        out_data = np.take_along_axis(self.data, picked, axis)
        if not keepdims:
            out_data = out_data.squeeze(axis)

        out = Tensor(out_data, (self,))

        def grad_fn():
            # d(max(a))/da = 1 for the max element, else 0
            g = out.grad if keepdims else np.expand_dims(out.grad, axis)
            mask = np.zeros_like(self.data)
            np.put_along_axis(mask, picked, 1.0, axis)
            self.grad += mask * g
        out.grad_fn = grad_fn

        return out

    def backward(self):
        ordered = []
        visited = set()

        # Build topological ordering of the graph
        def build(v):
            if v not in visited:
                visited.add(v)
                for inp in v.inputs:
                    build(inp)
                ordered.append(v)

        build(self)

        # Backpropagate through the graph in reverse order
        self.grad = np.ones_like(self.data)
        for v in reversed(ordered):
            v.grad_fn()




if __name__=="__main__":
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b + a
    c.backward()
    print(a.grad, b.grad)

    x = Tensor(np.random.rand(4, 3))
    W = Tensor(np.random.rand(3, 2))
    b = Tensor(np.zeros(2))

    loss = ((x @ W + b).relu() ** 2).mean()
    loss.backward()

    X = Tensor(np.arange(12, dtype=float).reshape(3, 4))
    y = (x.reshape(2, 6)) * Tensor(np.arange(12, dtype=float).reshape(2, 6)).sum()
    y.backward()
    print(x.grad)

    print(W.grad.shape, b.grad.shape)

