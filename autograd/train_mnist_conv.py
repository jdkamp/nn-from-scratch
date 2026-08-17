import numpy as np
from autograd.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Sequential, SoftMaxCrossEntropy
from data import load_mnist_splits
from baseline import accuracy
from autograd.train import train, predict

if __name__=="__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_splits(as_images=True)

    # Setup model
    rng = np.random.default_rng(0)
    model = Sequential(
        Conv2d(1, 8, 3, rng, padding=1),    # (N,1,28,28) -> (N,8,28,28)
        ReLU(),
        MaxPool2d(2),                       # -> (N,8,14,14)
        Conv2d(8, 16, 3, rng, padding=1),   # -> (N,16,14,14)
        ReLU(),
        MaxPool2d(2),                       # -> (N,16,7,7)
        Flatten(),                          # -> (N,784)
        Linear(16*7*7, 10, rng),            # -> (N,10)
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
        
