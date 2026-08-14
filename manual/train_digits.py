import numpy as np
from manual.nn import Linear, SoftmaxCrossEntropy, ReLU, Sequential
from data import load_digit_splits
from baseline import accuracy
from manual.train import train

if __name__=="__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_digit_splits()

    # Setup model
    rng = np.random.default_rng(0)
    model = Sequential(
        Linear(64, 32, rng),
        ReLU(),
        Linear(32, 10, rng)
    )
  
    # Training
    best_val, best_epoch = train(model, SoftmaxCrossEntropy(), accuracy,
                                 X_train, y_train, X_val, y_val,
                                 n_epochs=200, higher_is_better=True)


    # Evaluation
    most_common = np.bincount(y_train).argmax()
    baseline_acc = (y_val == most_common).mean()
    test_acc = accuracy(model.forward(X_test), y_test)

    print(f"\nbaseline   {baseline_acc * 100:.1f}%")
    print(f"best val   {best_val * 100:.1f}%  (epoch {best_epoch})")
    print(f"test       {test_acc * 100:.1f}%")
        
