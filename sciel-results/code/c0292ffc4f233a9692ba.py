"""Symbolic regression with hybrid polynomial and RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a hybrid model: degree-2 polynomial + 7 Gaussian RBFs.
    
    x: input array (values in [0.1, 6])
    params: array of length 10, coefficients for the basis functions
    
    Returns: array of predictions, one per point in x.
    """
    n = len(x)
    if n == 0:
        return np.array([])
        
    # Basis 1-3: Polynomial terms 1, x, x^2
    # Basis 4-10: 7 Gaussian RBFs
    
    # Centers for RBFs evenly spaced in [0.1, 6.0]
    x_min = 0.1
    x_max = 6.0
    num_rbf = 7
    centers = np.linspace(x_min + 0.2, x_max - 0.2, num_rbf)
    
    # Fixed width for RBFs
    sigma = 0.8
    
    # Build basis matrix B of shape (n, 10)
    B = np.zeros((n, 10), dtype=np.float64)
    
    # Polynomial part
    B[:, 0] = 1.0
    B[:, 1] = x
    B[:, 2] = x * x
    
    # RBF part
    # Compute distances from each x to each center
    # x is (n, 1), centers is (1, 7)
    x_col = x.reshape(-1, 1)
    centers_row = centers.reshape(1, -1)
    
    dist_sq = (x_col - centers_row) ** 2
    # Gaussian kernel
    rbf_values = np.exp(-dist_sq / (2.0 * sigma * sigma))
    
    # Assign to columns 3..9
    B[:, 3:10] = rbf_values
    
    # Linear combination
    return B @ params
# EVOLVE-BLOCK-END
