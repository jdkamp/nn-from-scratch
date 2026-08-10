import numpy as np
from nn import Linear, MSELoss, ReLU, Sequential
from data import load_splits
from baseline import rmse

if __name__=="__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Setup model
    rng = np.random.default_rng(0)
    model = Sequential(
        Linear(8, 64, rng),
        ReLU(),
        Linear(64, 1, rng)
    )
    loss_fn = MSELoss()

    # Training
    learning_rate = 0.1
    n_epochs = 50
    batch_size = 64
    n_samples = len(X_train)

    best_val = float("inf")
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

        train_rmse = rmse(model.forward(X_train), y_train)
        val_rmse = rmse(model.forward(X_val), y_val)

        if val_rmse < best_val: # early stopping
            best_val = val_rmse
            best_epoch = epoch
            best_params = [param.copy() for param, grad in model.parameters()]
        print(f"epoch {epoch:4d}    train {train_rmse:.4f}    val {val_rmse:.4f}")

    for (param, grad), saved in zip(model.parameters(), best_params):
        param[:] = saved

    # Evaluation
    final_val = rmse(model.forward(X_val), y_val)
    baseline_rmse = rmse(np.full_like(y_val, y_train.mean()), y_val)
    test_rmse = rmse(model.forward(X_test), y_test)
    print(f"\nbaseline  {baseline_rmse:.4f}   (${baseline_rmse * 100_000:,.0f})")
    print(f"model     {final_val:.4f}   (${final_val * 100_000:,.0f})")
    print(f"improvement: {(1 - final_val / baseline_rmse) * 100:.1f}%")
    print(f"test      {test_rmse:.4f}   (${test_rmse * 100_000:,.0f})")
        
