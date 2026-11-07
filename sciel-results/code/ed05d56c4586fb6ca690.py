"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Chebyshev polynomials to approximate the target.
    Maps x in [-2, 2] to t in [-1, 1] and uses T_0(t) to T_9(t).
    This basis is well-conditioned for BFGS optimization.
    """
    p = params
    
    # Ensure finite params to avoid NaN/Inf during optimization steps
    if not np.all(np.isfinite(p)):
        p = np.ones(10)

    # Map x from [-2, 2] to t in [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2t T_k(t) - T_{k-1}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # Linear combination: y = sum(p_i * T_i(t))
    # Using matrix multiplication for efficiency
    return T.T @ p
# EVOLVE-BLOCK-END
