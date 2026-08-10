import numpy as np
from nn import MSELoss, Linear
from data import load_splits

def compute_loss():
    return loss_fn.forward(layer.forward(X), y)

def numerical_grad(param, eps=1e-5):
    grad = np.zeros_like(param)

    for idx in np.ndindex(param.shape):
        original = param[idx]
        param[idx] = original + eps
        loss_plus = compute_loss()

        param[idx] = original - eps
        loss_minus = compute_loss()

        param[idx] = original     # restore param

        # Centered difference: error shrinks with eps^2
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

    return grad

if __name__=="__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    X = X_train[:5]
    y = y_train[:5]

    rng = np.random.default_rng(0)
    layer = Linear(8, 1, rng)
    loss_fn = MSELoss()

    compute_loss()
    layer.backward(loss_fn.backward())
    analytic_W = layer.dW.copy()    # copy: the next backward() would overwrite these
    analytic_b = layer.db.copy()

    numeric_W = numerical_grad(layer.W)
    numeric_b = numerical_grad(layer.b)

    # Relative error
    rel = np.abs(analytic_W - numeric_W) / np.maximum(1e-8, np.abs(analytic_W) + np.abs(numeric_W))
    print(f"W: {rel.max():.2e}")
    rel = np.abs(analytic_b - numeric_b) / np.maximum(1e-8, np.abs(analytic_b) + np.abs(numeric_b))
    print(f"b: {rel.max():.2e}")



