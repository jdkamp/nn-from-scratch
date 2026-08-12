import numpy as np
from data import load_housing_splits
from baseline import rmse

class Linear:
    """ Fully connected layer: y = x @ W + b """
    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):           # grad = dL/dy
        self.dW = self.x.T @ grad       # dL/dW = dy/dW x dL/dy
        self.db = grad.sum(axis=0)      # dL/db = dy/db x dL/dy
        return grad @ self.W.T          # dL/dx = dy/dx x dL/dy

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

class ReLU:
    """ Activation function - Rectified Linear Unit"""
    def forward(self, x):
        self.positive = x > 0
        return x * self.positive

    def backward(self, grad):
        return grad * self.positive

    def parameters(self):
        return []

class Sequential:
    """ Stack of layers """
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class MSELoss:
    """ Loss function """
    def forward(self, predictions, targets):
        self.diff = predictions - targets
        return float(np.mean(self.diff ** 2))
    
    def backward(self):
        return 2.0 * self.diff / self.diff.size # dL/dy
    
class SoftmaxCrossEntropy:
    """Softmax + cross entropy"""
    def forward(self, logits, targets):
        shifted = logits - logits.max(axis=1, keepdims=True) # avoid overflow
        exp = np.exp(shifted)                               # all positive
        self.probs = exp / exp.sum(axis=1, keepdims=True)   # each row sums to 1

        self.targets = targets
        n = len(targets)
        correct = self.probs[np.arange(n), targets]         # paired: row i, its true class
        return float(-np.log(correct).mean())               # low prob is punished the the most

    def backward(self):
        """ Gradient of the loss of the logits: p - onehot(y).
            Positive for wrong classes (push their logit down), negative for the
            single true class (push it up). Each row sums to 0.
        """
        n = len(self.targets)
        grad = self.probs.copy()                    # copy for -= in next line
        grad[np.arange(n), self.targets] -= 1.0     # p - onehot
        return grad / n                             # loss averaged over n samples



if __name__=="__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_housing_splits()

    # Build one layer
    rng = np.random.default_rng(0)
    layer = Linear(8, 1, rng)
    predictions = layer.forward(X_train)

    print(predictions.shape)
    print(rmse(predictions, y_train))

    # Check the loss and its gradient
    loss_fn = MSELoss()
    print(loss_fn.forward(predictions, y_train))
    grad = loss_fn.backward()
    print(grad.shape)

    # Check the backward pass
    dx = layer.backward(grad)
    print(layer.dW.shape)
    print(layer.db.shape)
    print(dx.shape)