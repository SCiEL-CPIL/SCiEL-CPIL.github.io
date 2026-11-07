"""Symbolic regression with 1 input: 10-parameter RBF with fixed centers."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Gaussian RBF network.
    
    Centers are fixed at evenly spaced points in [-2, 2].
    Widths are fixed (deterministic).
    params[0] is a global bias.
    params[1:10] are the weights for 9 RBF basis functions.
    
    This is a linear model in params, so BFGS will converge to the
    global optimum (linear least squares solution).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # 9 RBF centers evenly spaced in [-2, 2]
    # Leave one param for bias
    n_rbf = 9
    centers = np.linspace(-2.0, 2.0, n_rbf)
    
    # Fixed widths: choose a reasonable width
    # Spacing between centers is 4/(n_rbf-1) = 0.5
    # Use width ~ 0.4 for good overlap
    width = 0.4
    width_sq = width * width
    
    # Compute RBF basis values: shape (n, n_rbf)
    # basis[i, j] = exp(-(x[i] - centers[j])^2 / width_sq)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-diff * diff / width_sq)
    
    # params[0] is bias, params[1:10] are RBF weights
    bias = params[0]
    weights = params[1:1 + n_rbf]
    
    # prediction = bias + basis @ weights
    pred = bias + basis @ weights
    
    return pred
