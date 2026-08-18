import numpy as np
from autograd.engine import Tensor
from tests.helpers import check


rng = np.random.default_rng(0)

def test_add():
    a = Tensor(rng.normal(size=(3, 4)))
    b = Tensor(rng.normal(size=(4,)))
    check(lambda a, b: a + b, a, b)

def test_mul():
    a = Tensor(rng.normal(size=(3, 4)))
    b = Tensor(rng.normal(size=(4,)))
    check(lambda a, b: a * b, a, b)

def test_matmul():
    a = Tensor(rng.normal(size=(3, 4)))
    b = Tensor(rng.normal(size=(4, 2)))
    check(lambda a, b: a @ b, a, b)

def test_neg():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: -a, a)

def test_sub():
    a = Tensor(rng.normal(size=(3, 4)))
    b = Tensor(rng.normal(size=(4,)))
    check(lambda a, b: a - b, a, b)

def test_pow():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a ** 3, a)

def test_mean():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a.mean(), a)

def test_truediv():
    a = Tensor(rng.normal(size=(3, 4)))
    b = Tensor(rng.uniform(0.5, 2.0, size=(4,)))    # away from zero
    check(lambda a, b: a / b, a, b)

def test_exp():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a.exp(), a)

def test_relu():
    x = rng.normal(size=(3, 4))
    a = Tensor(x + np.sign(x) * 0.5)    # away from 0, where relu is not differentiable
    check(lambda a: a.relu(), a)

def test_log():
    a = Tensor(rng.uniform(0.5, 2.0, size=(3, 4)))  # log needs positive inputs
    check(lambda a: a.log(), a)

def test_sum():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a.sum(), a)

def test_sum_axis():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a.sum(axis=1), a)

def test_sum_axis_keepdims():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a.sum(axis=1, keepdims=True), a)

def test_reshape():
    a = Tensor(rng.normal(size=(2, 6)))
    check(lambda a: a.reshape(3, 4), a)

def test_transpose():
    a = Tensor(rng.normal(size=(2, 3, 4)))  # three distinct axis lengths
    check(lambda a: a.transpose(1, 2, 0), a)

def test_max():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a.max(axis=1), a)

def test_max_keepdims():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a.max(axis=1, keepdims=True), a)

def test_unfold():
    """unfold's backward calls fold, so this checks the adjoint numerically."""
    a = Tensor(rng.normal(size=(2, 3, 5, 5)))
    check(lambda a: a.unfold(2, 2), a)

def test_reuse():
    a = Tensor(rng.normal(size=(3, 4)))
    check(lambda a: a * a, a)
