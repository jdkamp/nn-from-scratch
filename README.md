# nn-from-scratch

A small neural network built with NumPy: layers, backpropagation and mini-batch SGD written by hand, no autograd framework.

## Files

- `data.py` - loaders for both datasets: split, standardize on training statistics, clip outliers
- `nn.py` - `Linear`, `ReLU`, `Sequential`, `MSELoss`, `SoftmaxCrossEntropy`: each with `forward` and `backward`
- `gradcheck.py` - verifies the analytic gradients against centered differences
- `baseline.py` - `rmse` and `accuracy`
- `train.py` - mini-batch SGD with early stopping
- `train_housing.py`, `train_digits.py` - the two entry points

## Run

```bash
pip install numpy scikit-learn
python train_housing.py
python train_digits.py
