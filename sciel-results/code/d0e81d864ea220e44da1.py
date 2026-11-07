"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a linear combination of Chebyshev polynomials of the first kind.
    This provides a stable, universal approximation for smooth functions
    on the interval [0.1, 6].
    """
    # Map x from [0.1, 6.0] to u in [-1, 1]
    x_min = 0.1
    x_max = 6.0
    center = (x_min + x_max) / 2.0
    scale = (x_max - x_min) / 2.0
    
    x_safe = np.clip(np.asarray(x, dtype=np.float64), x_min, x_max)
    u = (x_safe - center) / scale
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    basis = np.empty((len(x_safe), 10), dtype=np.float64)
    
    if len(x_safe) == 0:
        return np.empty(0)

    basis[:, 0] = 1.0
    basis[:, 1] = u
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * u * basis[:, k-1] - basis[:, k-2]
    
    # Linear combination
    return basis @ params

# EVOLVE-BLOCK-END
