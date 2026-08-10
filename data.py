from sklearn.datasets import fetch_california_housing
import numpy as np

def load_raw():
    """ Load data from sklearn datasets"""
    housing = fetch_california_housing()
    return housing.data, housing.target, housing.feature_names

def describe():
    """ Describes min / max / mean of features """
    X, y, names = load_raw()
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    means = X.mean(axis=0)

    print("Data set properties")
    print(f"{"feature":<12} {"min":>10} {"max":>10} {"mean":>10}")
    for name, lo, hi, avg in zip(names, mins, maxs, means):
        print(f"{name:>12} {lo:>10.2f} {hi:>10.2f} {avg:>10.2f}")

    print(f"\n{"target":<12} {y.min():>10.2f} {y.max():>10.2f} {y.mean():>10.2f} \n")
            

def load_splits(seed=0):
    # Load data
    X, y, names = load_raw()
    y = y.reshape(-1, 1)

    # Shuffle data
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # Slice data
    n_train = int(0.7 * len(X))
    n_val = int(0.85 * len(X))
    X_train, X_val, X_test = X[:n_train], X[n_train:n_val], X[n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_val], y[n_val:]

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)

    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma

    X_train = np.clip(X_train, -5.0, 5.0)
    X_val = np.clip(X_val, -5.0, 5.0)
    X_test = np.clip(X_test, -5.0, 5.0)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__=="__main__":
    describe()

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    print("Splits properties")
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    print(f"{"name":<9} {"shape":>10} {"min":>10} {"max":>10} {"mean":>10}")
    for name, arr in splits.items():
        print(f"{name:<9} {str(arr.shape):>10} {arr.min():>10.2f} {arr.max():>10.2f} {arr.mean():>10.2f}")


