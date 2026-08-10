from data import load_splits
import numpy as np

def rmse(predictions, targets):
    """ Root Mean Squared Error """
    return np.sqrt(np.mean((predictions - targets) ** 2))

if __name__=="__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Compute baseline
    baseline = y_train.mean()
    predictions = np.full_like(y_val, baseline)

    error = rmse(predictions, y_val)
    print(f"RMSE baseline: {error} - {error * 100_000:10.2f}$")