"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x to [-1, 1] using the actual min/max of the input data.
    # This ensures the Chebyshev basis is optimally scaled for the specific points,
    # improving numerical conditioning and fit accuracy.
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Handle edge case where all x are identical
    if x_max == x_min:
        return np.full_like(x, params[0], dtype=np.float64)
        
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Precompute Chebyshev polynomials T_0 to T_9
    T = np.zeros((len(x), 10), dtype=np.float64)
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
        
    # Compute linear combination
    return T @ params
# EVOLVE-BLOCK-END
