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

class Flatten:
    """Collapses the feature axes: (N, ax0, ax1, ax2, ...) -> (N, ax0*ax1*ax2*...)
    
    Bridges the convolutional Layer to Linear
    """
    def __call__(self, x):
        return x.reshape(x.data.shape[0], -1) # keep the batch dimension, flatten the rest

    def parameters(self):
        return[]


class Conv2d:
    """Convolution Layer - Slide out_channels learned filters over a batch of images.

    Input  (batch_size, in_channels, in_h, in_w)
    Output (batch_size, out_channels, out_h, out_w) - out_h/w = (in_h/w + 2*padding - k) // stride + 1

    k        size of the square kernel
    stride   step between sliding windows
    padding  zeros added around the border
    rng      random number generator (for weight initialization)
    """
    def __init__(self, in_channels, out_channels, k, rng, stride=1, padding=0):
        scale = np.sqrt(2.0 / (in_channels * k * k))
        self.W = Tensor(rng.normal(0.0, scale, (out_channels, in_channels, k, k)))
        self.b = Tensor(np.zeros(out_channels))
        self.k, self.stride, self.padding = k, stride, padding

    def __call__(self, x):
        batch_size, _ , in_h, in_w = x.data.shape
        k, s, p = self.k, self.stride, self.padding
        out_h = (in_h + 2 * p - k) // s + 1
        out_w = (in_w + 2 * p - k) // s + 1

        cols = x.unfold(k, k, s, p)
        w_col = self.W.reshape(self.W.data.shape[0], -1).transpose(1, 0)
        out = cols @ w_col + self.b

        return out.reshape(batch_size, out_h, out_w, -1).transpose(0, 3, 1, 2)

    def parameters(self):
        return [self.W, self.b]

class MaxPool2d:
    """Max Pooling Layer - Slide a window over the input and take the max value"""
    def __init__(self, k, stride=None):
        self.k = k
        self.stride = stride or k

    def __call__(self, x):
        batch_size, n_channels, in_h, in_w = x.data.shape
        k, s = self.k, self.stride
        out_h = (in_h - k) // s + 1
        out_w = (in_w - k) // s + 1

        cols = x.reshape(batch_size*n_channels, 1, in_h, in_w).unfold(k, k, s, 0)
        pooled = cols.max(axis=1)
        return pooled.reshape(batch_size, n_channels, out_h, out_w)

    def parameters(self):
        return []


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