"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import chebyshev_t


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a linear combination of Chebyshev polynomials.
    
    Args:
        x: Input array of shape (N,)
        params: Parameters of shape (10,)
    
    Returns:
        Predictions of shape (N,)
    """
    # Ensure params is finite
    if not np.all(np.isfinite(params)):
        params = np.ones(10)
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T is shape (N, 10)
    N = len(x)
    T = np.zeros((N, 10))
    
    if N == 0:
        return np.array([])
    
    # T_0(t) = 1
    T[:, 0] = 1.0
    
    # T_1(t) = t
    T[:, 1] = t
    
    # Recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Return linear combination
    return T @ params
# EVOLVE-BLOCK-END
