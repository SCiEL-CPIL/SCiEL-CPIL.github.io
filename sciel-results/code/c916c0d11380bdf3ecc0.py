"""Rewritten program using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a 9th-degree Chebyshev polynomial series.
    
    The input x is mapped from [-2, 2] to [-1, 1] to utilize the standard 
    Chebyshev polynomials of the first kind, which are well-conditioned 
    on this interval.
    
    y(x) = sum_{k=0}^{9} params[k] * T_k(x/2)
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    # Map x to [-1, 1]
    t = x / 2.0
    
    # Initialize the basis matrix
    # Shape: (n, 10)
    # Column k contains T_k(t)
    T = np.empty((n, 10), dtype=float)
    
    # T_0(t) = 1
    T[:, 0] = 1.0
    
    # T_1(t) = t
    T[:, 1] = t
    
    # Recurrence relation for k >= 2:
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    # We need T_2 to T_9
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
        
    # Linear combination of basis functions with params
    return T @ params
# EVOLVE-BLOCK-END
