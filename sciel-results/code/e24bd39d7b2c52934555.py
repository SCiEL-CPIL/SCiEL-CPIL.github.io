"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a Chebyshev polynomial basis.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if not _check_deadline():
        # Return a simple fallback if we're running out of time
        return np.ones_like(x) * params[0]
    
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2x - (max + min)) / (max - min)
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Clip t to [-1, 1] for numerical stability
    t = np.clip(t, -1.0, 1.0)
    
    # Number of Chebyshev terms
    n_terms = 10
    
    # Compute Chebyshev polynomials T_k(t) for k=0..9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2t T_k(t) - T_{k-1}(t)
    
    basis = np.empty((len(x), n_terms), dtype=np.float64)
    
    if n_terms > 0:
        basis[:, 0] = 1.0
        
    if n_terms > 1:
        basis[:, 1] = t
        
    for k in range(2, n_terms):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Check for non-finite values and handle gracefully
    if not np.all(np.isfinite(basis)):
        # Fallback: use only the first few terms that are finite
        for col in range(n_terms):
            if not np.all(np.isfinite(basis[:, col])):
                basis[:, col:] = 0.0
                basis[:, col] = 0.0
                break
    
    # Prediction is the linear combination of the basis functions
    result = basis @ params
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
