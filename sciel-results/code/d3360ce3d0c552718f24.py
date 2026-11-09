"""Symbolic regression with 1 input(s) using a hybrid RBF + Chebyshev basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t_start = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    
    # Map x from [0.1, 6] to t in [-1, 1] for Chebyshev stability
    t = (2.0 * x - 6.1) / 5.9
    
    # Compute Chebyshev polynomials T_0 to T_4 (5 terms)
    basis_cheb = np.zeros((n, 5), dtype=np.float64)
    basis_cheb[:, 0] = 1.0
    basis_cheb[:, 1] = t
    for k in range(2, 5):
        basis_cheb[:, k] = 2.0 * t * basis_cheb[:, k-1] - basis_cheb[:, k-2]
    
    # RBF basis with 5 Gaussian terms
    # Centers spread across the domain [0.1, 6]
    centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bandwidths = np.array([0.8, 1.0, 1.2, 1.0, 0.8])
    
    basis_rbf = np.zeros((n, 5), dtype=np.float64)
    for i in range(5):
        basis_rbf[:, i] = np.exp(-((x - centers[i])**2) / (2.0 * bandwidths[i]**2))
    
    # Combine into a single basis matrix of shape (n, 10)
    basis = np.hstack([basis_cheb, basis_rbf])
    
    # Linear combination: prediction = basis @ params
    pred = basis @ params
    
    # Ensure finite values
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
