import numpy as np

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
            X_batch, y_batch = X_train[idx], y_train[idx]

            predictions = model.forward(X_batch)            # forward
            loss = loss_fn.forward(predictions, y_batch)    # measure
            model.backward(loss_fn.backward())              # gradient

            for param, grad in model.parameters():
                param -= learning_rate * grad               # update parameters

        train_score = metric(model.forward(X_train), y_train)
        val_score = metric(model.forward(X_val), y_val)

        improved = val_score > best_val if higher_is_better else val_score < best_val
        if improved: # early stopping
            best_val = val_score
            best_epoch = epoch
            best_params = [param.copy() for param, grad in model.parameters()]
        print(f"epoch {epoch:4d}    train {train_score:.4f}    val {val_score:.4f}")

    for (param, grad), saved in zip(model.parameters(), best_params):
        param[:] = saved

    return best_val, best_epoch
