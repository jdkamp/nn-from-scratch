import numpy as np
from autograd.nn import Linear, MSELoss, ReLU, Sequential
from data import load_housing_splits
from baseline import rmse
from autograd.train import train
from autograd.engine import Tensor

if __name__=="__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_housing_splits()

    # Setup model
    rng = np.random.default_rng(0)
    model = Sequential(
        Linear(8, 64, rng),
        ReLU(),
        Linear(64, 1, rng)
    )
    
    # Training
    best_val, best_epoch = train(model, MSELoss(), rmse,
                                    X_train, y_train, X_val, y_val,
                                    n_epochs=50, higher_is_better=False)
    
    

    # Evaluation
    baseline_rmse = rmse(np.full_like(y_val, y_train.mean()), y_val)
    test_rmse = rmse(model(Tensor(X_test)).data, y_test)

    print(f"\nbaseline   {baseline_rmse:.4f}   (${baseline_rmse * 100_000:,.0f})")
    print(f"best val   {best_val:.4f}   (${best_val * 100_000:,.0f})  (epoch {best_epoch})")
    print(f"test       {test_rmse:.4f}   (${test_rmse * 100_000:,.0f})")
    print(f"improvement over baseline: {(1 - best_val / baseline_rmse) * 100:.1f}%")
        
