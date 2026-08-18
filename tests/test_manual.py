import numpy as np
from manual.nn import Linear, ReLU, Sequential, MSELoss, SoftmaxCrossEntropy
from tests.helpers import numerical_grad

rng = np.random.default_rng(0)


def check_manual(model, loss_fn, X, y, rtol=1e-5, atol=1e-7):
    """Assert the hand-derived gradients match centered differences.

    The manual layers keep their gradients alongside the parameters, so
    parameters() yields (param, grad) pairs rather than Tensors.
    """
    def compute_loss():
        return loss_fn.forward(model.forward(X), y)

    compute_loss()  # populates the caches backward() needs
    model.backward(loss_fn.backward())

    for param, grad in model.parameters():
        analytic = grad.copy()  # copy: numerical_grad mutates param in place
        numeric = numerical_grad(param, compute_loss)
        np.testing.assert_allclose(analytic, numeric, rtol=rtol, atol=atol)


def test_mse_gradients():
    model = Sequential(Linear(4, 3, rng), ReLU(), Linear(3, 1, rng))
    X = rng.normal(size=(5, 4))
    y = rng.normal(size=(5, 1))
    check_manual(model, MSELoss(), X, y)


def test_softmax_cross_entropy_gradients():
    model = Sequential(Linear(4, 3, rng), ReLU(), Linear(3, 6, rng))
    X = rng.normal(size=(5, 4))
    y = rng.integers(0, 6, size=5)  # class labels, 6 classes
    check_manual(model, SoftmaxCrossEntropy(), X, y)
