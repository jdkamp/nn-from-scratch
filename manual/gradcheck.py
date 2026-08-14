import numpy as np
from manual.nn import ReLU, Linear, Sequential, MSELoss, SoftmaxCrossEntropy
from data import load_housing_splits, load_digit_splits

def numerical_grad(param, compute_loss, eps=1e-5):
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

def check(name, model, loss_fn, X, y):
    # Closure over this call's model and data, so check() works for any of them
    def compute_loss():
        return loss_fn.forward(model.forward(X), y)

    compute_loss()
    model.backward(loss_fn.backward())

    print(name)
    for i, (param, grad) in enumerate(model.parameters()):
        analytic = grad.copy()      # copy before numerical_grad mutates param
        numeric = numerical_grad(param, compute_loss)

        # Relative error
        rel = np.abs(analytic - numeric) / np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))
        print(f"  param {i}: {rel.max():.2e}")

if __name__=="__main__":
    rng = np.random.default_rng(0)

    X_train, X_val, X_test, y_train, y_val, y_test = load_housing_splits()
    check("housing (MSE)",
          Sequential(Linear(8, 4, rng), ReLU(), Linear(4, 1, rng)),
          MSELoss(), X_train[:5], y_train[:5])

    X_train, X_val, X_test, y_train, y_val, y_test = load_digit_splits()
    check("digits (softmax + cross entropy)",
          Sequential(Linear(64, 4, rng), ReLU(), Linear(4, 10, rng)),
          SoftmaxCrossEntropy(), X_train[:5], y_train[:5])
