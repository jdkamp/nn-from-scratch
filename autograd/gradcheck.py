import numpy as np
from autograd.nn import Conv2d
from autograd.engine import Tensor

rng = np.random.default_rng(0)
conv = Conv2d(1, 8, 3, rng, padding=1)
out = conv(Tensor(np.random.rand(4, 1, 8, 8)))
print(out.data.shape)                    # (4, 8, 8, 8)

out.sum().backward()
print(conv.W.grad.shape, conv.b.grad.shape)   # (8,1,3,3) and (8,)
