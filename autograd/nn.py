import numpy as np
from autograd.engine import Tensor

class Linear:
    def __init__(self, n_in, n_out, rng):
        self.W = Tensor(rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out)))
        self.b = Tensor(np.zeros(n_out))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

class ReLU:
    def __call__(self, x):
        return x.relu()

    def parameters(self):
        return []

class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class MSELoss:
    def __call__(self, predictions, targets):
        return ((predictions - targets) ** 2).mean()

class SoftMaxCrossEntropy:
    def __call__(self, logits, targets):
        t = targets.data.astype(int).ravel()
        n = len(t)

        shift = logits.data.max(axis=1, keepdims=True)
        z = logits - Tensor(shift)

        log_probs = z - z.exp().sum(axis=1, keepdims=True).log()

        onehot = np.zeros_like(logits.data)
        onehot[np.arange(n), t] = 1.0

        return -(Tensor(onehot) * log_probs).sum() * Tensor(1.0 / n)