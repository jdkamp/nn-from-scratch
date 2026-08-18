import numpy as np

def numerical_grad(param, compute_loss, eps=1e-5):
    """Estimate d(loss)/d(param) by changing one element at a time.

    param         mutated in place while measuring, and restored afterwards
    compute_loss  re-reads param and returns the loss as a single number
    eps           step size
    """
    grad = np.zeros_like(param)

    for idx in np.ndindex(param.shape):
        original = param[idx]
        param[idx] = original + eps # mutates the param in place
        loss_plus = compute_loss()  # reads the param

        param[idx] = original - eps # mutates the original in place
        loss_minus = compute_loss() # reads the param

        param[idx] = original     # restore param

        # Centered difference: error shrinks with eps^2
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

    return grad

def check(f, *tensors, rtol=1e-5, atol=1e-7, seed=0):
    """Assert the gradients from backward() match centered differences.

    f           builds the computation and returns a Tensor of any shape.
    *tensors    the tensors whose gradients are checked.
    rtol, atol  tolerances
    seed        for the probe below

    Differentiation needs a scalar output (like a loss), therefore f's result is
    summed. A random probe is additional needed because an operation like  transpose
    or reshape leaves  the sum unchanged, and a wrong backward rule would go unnoticed.
    """
    probe = np.random.default_rng(seed).normal(size=f(*tensors).data.shape)

    def loss():
        return (f(*tensors) * probe).sum()  # Scalar loss

    for t in tensors:
        t.zero_grad()       # gradients accumulate with +=, so start from zero
    loss().backward()

    for t in tensors:
        analytic = t.grad.copy()    # copy: numerical_grad mutates t.data in place
        numeric = numerical_grad(t.data, lambda: loss().data)
        np.testing.assert_allclose(analytic, numeric, rtol=rtol, atol=atol)
