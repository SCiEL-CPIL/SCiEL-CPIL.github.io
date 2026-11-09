"""Symbolic regression using a Radial Basis Function (RBF) network with fixed centers and widths."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Sum of 10 Gaussian RBFs.
    y(x) = sum_{i=0}^{9} params[i] * exp(-((x - c_i)^2) / (2 * sigma^2))
    
    Centers c_i are fixed to be evenly spaced in [-2, 2].
    Width sigma is fixed.
    params are the linear coefficients.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define fixed centers and width
    # 10 centers evenly spaced from -2 to 2
    centers = np.linspace(-2.0, 2.0, 10)
    sigma = 0.35
    
    # Compute RBF values: shape (N, 10)
    # x is (N,), centers is (10,)
    # dx is (N, 10)
    dx = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF matrix
    rbf = np.exp(-0.5 * (dx / sigma) ** 2)
    
    # Linear combination
    # rbf is (N, 10), params is (10,)
    # result is (N,)
    result = rbf @ params
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
