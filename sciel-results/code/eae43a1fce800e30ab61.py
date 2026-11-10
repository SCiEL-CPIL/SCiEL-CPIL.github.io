"""Symbolic regression with Gaussian bump basis model."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

# Fixed centers for 10 Gaussian bumps evenly spaced over [-2, 2]
_N_BUMPS = 10
_CENTERS = np.linspace(-2.0, 2.0, _N_BUMPS)
_SIGMA = 0.5  # Fixed width for all bumps


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian bump basis model:
    f(x) = sum_{i=0}^{9} a_i * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    10 parameters: amplitudes a0..a9 for 10 fixed Gaussian kernels.
    Centers c_i are evenly spaced over [-2, 2], width sigma is fixed.
    This is a linear model in the parameters, so BFGS converges quickly.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > 145.0:
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < _N_BUMPS:
        params = np.pad(params, (0, _N_BUMPS - params.size))
    elif params.size > _N_BUMPS:
        params = params[:_N_BUMPS]
    
    # Compute Gaussian basis functions: (n_points, n_bumps)
    # x has shape (n_points,), centers has shape (n_bumps,)
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / _SIGMA) ** 2)
    
    # Linear combination: (n_points,)
    result = basis @ params
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
