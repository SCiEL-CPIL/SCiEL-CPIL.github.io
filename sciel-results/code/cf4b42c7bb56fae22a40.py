import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a degree-9 Chebyshev polynomial in x.
    
    x: input array (values in [0.1, 6])
    params: array of length 10, coefficients for T_0 through T_9
    
    Returns: array of predictions, one per point in x.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev basis
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Build Chebyshev basis matrix T of shape (n, 10)
    # T[:, k] = T_k(t) for k = 0, ..., 9
    T = np.zeros((n, 10), dtype=np.float64)
    T[:, 0] = 1.0
    if n > 0:
        T[:, 1] = t
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Compute predictions: T @ params
    return T @ params
