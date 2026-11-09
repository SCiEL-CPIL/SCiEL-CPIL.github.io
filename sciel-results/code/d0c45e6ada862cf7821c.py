# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial approximation of degree 9 on [0.1, 6].
    Maps x to [-1, 1] and evaluates T_0..T_9 with 10 parameters.
    This provides a smooth, well-conditioned basis for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1]
    # center = (0.1 + 6) / 2 = 3.05
    # half_range = (6 - 0.1) / 2 = 2.95
    u = (x - 3.05) / 2.95
    
    # Clip u to [-1, 1] for numerical stability, though inputs are in [0.1, 6]
    u = np.clip(u, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0..T_9 using recurrence:
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    N = x.shape[0]
    basis = np.empty((10, N), dtype=np.float64)
    basis[0, :] = 1.0
    basis[1, :] = u
    for k in range(2, 10):
        basis[k, :] = 2.0 * u * basis[k-1, :] - basis[k-2, :]
    
    # Linear combination
    y = p @ basis
    
    return y
# EVOLVE-BLOCK-END
