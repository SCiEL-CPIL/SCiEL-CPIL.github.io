"""Symbolic regression with 1 input(s) using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [0.1, 6] to t in [-1, 1] for Chebyshev stability
    # t = (2*x - (min+max)) / (max-min)
    # min=0.1, max=6.0
    # t = (2*x - 6.1) / 5.9
    
    x = np.asarray(x, dtype=np.float64)
    
    # Handle potential edge cases or ensure finite
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    t = (2.0 * x - 6.1) / 5.9
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n}(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # Initialize array for basis functions
    # shape: (len(x), 10)
    basis = np.zeros((x.shape[0], 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    if x.shape[0] > 0:
        basis[:, 1] = t
        
        for k in range(2, 10):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    # Linear combination
    # params is shape (10,)
    # prediction = basis @ params
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
