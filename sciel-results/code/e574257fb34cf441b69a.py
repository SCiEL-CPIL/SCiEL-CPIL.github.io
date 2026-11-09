"""Symbolic regression with Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian Radial Basis Function (RBF) basis.
    
    Basis functions (10 total):
    Gaussian RBFs centered at 10 points evenly spaced in [-3, 3].
    Each RBF is exp(-(x - c_i)^2 / (2 * sigma^2)).
    
    This basis is universal approximator and provides local flexibility,
    which may capture features that global polynomial bases miss or handle
    differently in the optimization landscape.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    n = x.shape[0]
    
    # Define 10 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width for the Gaussian kernels
    sigma = 1.0
    sigma2 = 2.0 * sigma * sigma
    
    # Compute RBF values
    # x is (n,), centers is (10,)
    # diff is (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / sigma2)
    
    # Combine with parameters
    # rbf_values is (n, 10), params is (10,)
    result = rbf_values @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
