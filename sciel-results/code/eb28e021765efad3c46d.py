"""Symbolic regression with polynomial basis on normalized input."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial model of degree 8 on normalized input.
    params[0:9] are coefficients for powers 0..8 of x_norm.
    params[9] is unused (kept for interface compatibility).
    x_norm = (x - 0.1) / 5.9  maps [0.1, 6] -> [0, 1]
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Normalize input to [0, 1]
    x_norm = (x - 0.1) / 5.9
    
    # Build polynomial: sum_{k=0}^{8} p[k] * x_norm^k
    # Use Horner's method for numerical stability
    result = np.full_like(x, params[8], dtype=np.float64)
    for k in range(7, -1, -1):
        result = result * x_norm + params[k]
    
    return result
# EVOLVE-BLOCK-END
