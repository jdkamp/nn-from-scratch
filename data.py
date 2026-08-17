from sklearn.datasets import fetch_california_housing, load_digits, fetch_openml
import numpy as np



def describe(X, y, features):
    """ Describes min / max / mean of features """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    means = X.mean(axis=0)

    print("Data set properties")
    print(f"{"feature":<12} {"min":>10} {"max":>10} {"mean":>10}")
    for name, lo, hi, avg in zip(features, mins, maxs, means):
        print(f"{name:>12} {lo:>10.2f} {hi:>10.2f} {avg:>10.2f}")

    print(f"\n{"target":<12} {y.min():>10.2f} {y.max():>10.2f} {y.mean():>10.2f} \n")
            

def shuffle_and_split(X, y, seed=0, train_frac=0.7, val_frac=0.85):
    """Shuffle rows, then split into train / val / test."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    n_train = int(train_frac * len(X))
    n_val = int(val_frac * len(X))
    return (X[:n_train], X[n_train:n_val], X[n_val:],
            y[:n_train], y[n_train:n_val], y[n_val:])

def standardize(X_train, X_val, X_test, clip=5.0):
    """Zero mean, unit std, then clip."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0             # a constant feature would divide by zero

    return tuple(np.clip((X - mu) / sigma, -clip, clip)
                 for X in (X_train, X_val, X_test))

def load_housing_splits(seed=0):
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    y = y.reshape(-1, 1)                # (N,) -> (N,1) to match the model output

    X_train, X_val, X_test, y_train, y_val, y_test = shuffle_and_split(X, y, seed)
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_digit_splits(seed=0, as_images=True):
    digits = load_digits()
    # Pixels are 0-16. Scaled, not standardized: some pixels are 0 in every
    # image, so sigma would be 0. Constant divisor, so it can precede the split.
    X = digits.data / 16.0
    y = digits.target

    if as_images:
        X = X.reshape(-1, 1, 8, 8)  # -1 - number of images, 1 - grey scale, 8, 8 - height, width

    return shuffle_and_split(X, y, seed)

def load_mnist_splits(seed=0, as_images=True):
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    X = np.concatenate([x_train, x_test]).astype(float) / 255.0    # combine + scale
    y = np.concatenate([y_train, y_test]).astype(int)

    X = X.reshape(-1, 1, 28, 28) if as_images else X.reshape(len(X), -1)

    return shuffle_and_split(X, y, seed) 




if __name__=="__main__":
    housing = fetch_california_housing()
    describe(housing.data, housing.target, housing.feature_names)

    X_train, X_val, X_test, y_train, y_val, y_test = load_housing_splits()

    print("Splits properties")
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    print(f"{"name":<9} {"shape":>10} {"min":>10} {"max":>10} {"mean":>10}")
    for name, arr in splits.items():
        print(f"{name:<9} {str(arr.shape):>10} {arr.min():>10.2f} {arr.max():>10.2f} {arr.mean():>10.2f}")

    #X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_splits()
    #print(X_train.shape, X_train.min(), X_train.max(), y_train[:10])
    load_mnist_splits()

