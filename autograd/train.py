import numpy as np
from autograd.engine import Tensor

def train(model, loss_fn, metric, X_train, y_train, X_val, y_val, learning_rate=0.1,
          n_epochs=50, batch_size=64, higher_is_better=False, seed=0):
    """Mini batch SDG with early stopping"""

    rng = np.random.default_rng(seed)
    n_samples = len(X_train)
    best_val = -np.inf if higher_is_better else np.inf
    best_epoch = -1
    best_params = None

    for epoch in range(n_epochs):
        perm = rng.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            X_batch, y_batch = X_train[idx], y_train[idx]   # Mini-batch

            predictions = model(Tensor(X_batch))            # forward pass
            loss = loss_fn(predictions, Tensor(y_batch))    # measure loss
            loss.backward()                                 # backward pass

            for param in model.parameters():
                param.data -= learning_rate * param.grad    # gradient descent step

            model.zero_grad()                               # reset gradients

        # Training and validation scores
        train_score = metric(model(Tensor(X_train)).data, y_train)
        val_score = metric(model(Tensor(X_val)).data, y_val)

        # Early stopping: save best model parameters
        improved = val_score > best_val if higher_is_better else val_score < best_val
        if improved:
            best_val = val_score
            best_epoch = epoch
            best_params = [param.data.copy() for param in model.parameters()]
        print(f"epoch {epoch:4d}    train {train_score:.4f}    val {val_score:.4f}")

    # Restore best model parameters
    for param, saved in zip(model.parameters(), best_params):
        param.data[:] = saved

    return best_val, best_epoch
