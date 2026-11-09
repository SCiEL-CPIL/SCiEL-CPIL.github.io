import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Convert to float64 array
    x = np.asarray(x, dtype=np.float64)
    
    # Handle potential edge cases
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    
    # Deadline guard
    t0 = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    # Hybrid basis: 5 RBFs + 5 Chebyshev terms = 10 total
    
    # Part 1: RBF basis (5 Gaussian kernels)
    # Centers spread across [0.1, 6]
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])[:5]
    # Fixed bandwidths - moderate width to capture local features
    bandwidths = np.array([0.8, 0.9, 1.0, 0.9, 0.8])
    
    rbf_basis = np.zeros((n, 5), dtype=np.float64)
    for j in range(5):
        diff = x - centers[j]
        rbf_basis[:, j] = np.exp(-(diff ** 2) / (2.0 * bandwidths[j] ** 2))
    
    # Part 2: Chebyshev basis (5 terms)
    # Map x from [0.1, 6] to t in [-1, 1]
    t = (2.0 * x - 6.1) / 5.9
    
    cheb_basis = np.zeros((n, 5), dtype=np.float64)
    cheb_basis[:, 0] = 1.0
    cheb_basis[:, 1] = t
    
    for k in range(2, 5):
        cheb_basis[:, k] = 2.0 * t * cheb_basis[:, k-1] - cheb_basis[:, k-2]
    
    # Combine: [RBF_1, ..., RBF_5, Cheb_0, ..., Cheb_4]
    basis = np.hstack([rbf_basis, cheb_basis])
    
    # Linear combination
    pred = basis @ params
    
    # Ensure finite values
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
