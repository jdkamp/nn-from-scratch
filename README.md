# nn-from-scratch

A small neural network built with NumPy: layers, backpropagation and mini-batch SGD written by hand, using manual and automatic differentiation.

## Files

Shared:

- `data.py` - loaders for datasets: split, standardize on training statistics, clip outliers
- `baseline.py` - `rmse` and `accuracy`

`manual/` - all gradients derived manually:
- `nn.py` - `Linear`, `ReLU`, `Sequential`, `MSELoss`, `SoftmaxCrossEntropy`: each with `forward` and `backward`
- `train.py` - mini-batch SGD with early stopping
- `train_housing.py`, `train_digits.py` - the entry points

`autograd/` - gradients derived automatically:
- `engine.py` - `Tensor` records a graph of operations
- `nn.py` - the same layers, **forward only** - no `backward` methods
- `conv_utils.py` - utility functions for conv net training
- `train.py`, `train_housing.py`, `train_digits.py`, `train_mnist.py`
- `train_digits_conv.py`, `train_mnist_conv.py` - training with conv nets

`tests/`- pytests

## Run

```bash
pip install numpy scikit-learn tensorflow # scikit-learn and tensorflow are only used to fetch the training data
python -m manual.train_housing          # regression
python -m manual.train_digits           # 8x8 digits
python -m autograd.train_housing        # regression with autograd
python -m autograd.train_digits         # 8x8 digits with autograd
python -m autograd.train_digits_conv    # conv net, 8x8 digits with autograd
python -m autograd.train_mnist          # MLP, 28x28 MNIST
python -m autograd.train_mnist_conv     # conv net, 28x28 MNIST with autograd
```

## References

- [micrograd](https://github.com/karpathy/micrograd) - the autograd engine follows its design: per-node backward closures and a topological walk, here over NumPy arrays
- Datasets: California housing and digits via scikit-learn; MNIST via `tensorflow.keras.datasets`
