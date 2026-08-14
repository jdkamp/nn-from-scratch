# nn-from-scratch

A small neural network built with NumPy: layers, backpropagation and mini-batch SGD written by hand, using manual and automatic differentiation.

## Files

Shared:

- `data.py` - loaders for both datasets: split, standardize on training statistics, clip outliers
- `baseline.py` - `rmse` and `accuracy`

`manual/` - all gradients derived manually:
- `nn.py` - `Linear`, `ReLU`, `Sequential`, `MSELoss`, `SoftmaxCrossEntropy`: each with `forward` and `backward`
- `train.py` - mini-batch SGD with early stopping
- `gradcheck.py` - verifies the analytic gradients against centered differences
- `train_housing.py`, `train_digits.py` - the two entry points

`autograd/` - gradients derived automatically:
- `engine.py` - `Tensor` records a graph of operations
- `nn.py` - the same layers, **forward only** - no `backward` methods
- `train.py`, `train_housing.py`, `train_digits.py`

## Run

```bash
pip install numpy scikit-learn
python -m manual.train_housing
python -m manual.train_digits
python -m autograd.train_housing
python -m autograd.train_digits
```

