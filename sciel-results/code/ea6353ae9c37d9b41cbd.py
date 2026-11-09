"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (degrees 0-4) for global trend
    + 5 Gaussian RBFs with fixed centers and varying bandwidths for local detail.
    
    x: input array of shape (N,)
    params: array of shape (10,)
      params[0:5]  -> Chebyshev coefficients
      params[5:10] -> RBF amplitude coefficients
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Normalize x to [-1, 1] for Chebyshev (x in [0.1, 6])
    x_min = 0.1
    x_max = 6.0
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    
    # Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(x_norm)
    T1 = x_norm.copy()
    T2 = 2.0 * x_norm**2 - 1.0
    T3 = 4.0 * x_norm**3 - 3.0 * x_norm
    T4 = 8.0 * x_norm**4 - 8.0 * x_norm**2 + 1.0
    
    cheb = np.column_stack([T0, T1, T2, T3, T4])  # shape (N, 5)
    
    # Gaussian RBFs: 5 fixed centers spread across [0.1, 6], varying bandwidths
    centers = np.array([0.5, 1.5, 3.0, 4.5, 5.5])
    bandwidths = np.array([0.5, 0.8, 1.2, 0.7, 0.6])  # sigma values
    
    rbf_terms = np.empty((len(x), 5), dtype=np.float64)
    for i in range(5):
        diff = x - centers[i]
        rbf_terms[:, i] = np.exp(-0.5 * (diff / bandwidths[i])**2)
    
    # Combine: prediction = cheb @ params[0:5] + rbf @ params[5:10]
    pred = cheb @ params[0:5] + rbf_terms @ params[5:10]
    
    return pred
# EVOLVE-BLOCK-END
