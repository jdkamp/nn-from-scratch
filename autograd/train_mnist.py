import numpy as np
from autograd.nn import Linear, ReLU, Sequential, SoftMaxCrossEntropy
from data import load_mnist_splits
from baseline import accuracy
from autograd.train import train, predict
from autograd.engine import Tensor

if __name__=="__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_splits(as_images=False)

    # Setup model
    rng = np.random.default_rng(0)
    model = Sequential(
        Linear(784, 128, rng),  # (N,784) -> (N,128)
        ReLU(),
        Linear(128, 10, rng)    # -> (N,10)
    )

    
    # Training
    best_val, best_epoch = train(model, SoftMaxCrossEntropy(), accuracy,
                                    X_train, y_train, X_val, y_val,
                                    n_epochs=15, higher_is_better=True)
    
    

    # Evaluation
    most_common = np.bincount(y_train).argmax()
    baseline_acc = (y_val == most_common).mean()
    test_acc = accuracy(predict(model, X_test), y_test)

    print(f"\nbaseline   {baseline_acc * 100:.1f}%")
    print(f"best val   {best_val * 100:.1f}%  (epoch {best_epoch})")
    print(f"test       {test_acc * 100:.1f}%")
        
