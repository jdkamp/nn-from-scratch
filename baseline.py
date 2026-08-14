from data import load_housing_splits
import numpy as np

def rmse(predictions, targets):
    """ Root Mean Squared Error """
    return np.sqrt(np.mean((predictions - targets) ** 2))

def accuracy(logits, targets):
    """Fraction of samples whose highest-scoring class is the correct one"""
    return (logits.argmax(axis=1) == targets).mean()

