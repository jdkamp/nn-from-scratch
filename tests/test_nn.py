import numpy as np
from autograd.engine import Tensor
from autograd.nn import Linear, Conv2d, MaxPool2d
from tests.helpers import check

rng = np.random.default_rng(0)

def test_linear():
    layer = Linear(4, 3, rng)
    x = Tensor(rng.normal(size=(5, 4)))
    check(lambda x, W, b: layer(x), x, layer.W, layer.b)

def test_conv2d():
    conv = Conv2d(2, 3, 3, rng, padding=1)
    x = Tensor(rng.normal(size=(2, 2, 5, 5)))
    check(lambda x, W, b: conv(x), x, conv.W, conv.b)

def test_conv2d_stride():
    conv = Conv2d(2, 3, 3, rng, stride=2)
    x = Tensor(rng.normal(size=(2, 2, 5, 5)))
    check(lambda x, W, b: conv(x), x, conv.W, conv.b)

def test_maxpool2d():
    pool = MaxPool2d(2)
    x = Tensor(rng.normal(size=(2, 3, 4, 4)))
    check(lambda x: pool(x), x)
