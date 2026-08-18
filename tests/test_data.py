import numpy as np
from data import shuffle_and_split

def test_split_and_shuffle():
    X = np.arange(100, dtype=float).reshape(100, 1)
    y = X[:, 0].copy()  # labels are the data's value

    X_train, X_val , X_test , y_train, _ , _ = shuffle_and_split(X, y)

    # data and labels are not changed
    np.testing.assert_array_equal(X_train[:, 0], y_train)

    # no data was lost
    X_all = np.concatenate([X_train[:, 0], X_val[:, 0], X_test[:, 0]])
    np.testing.assert_array_equal(np.sort(X_all), X[:, 0])

