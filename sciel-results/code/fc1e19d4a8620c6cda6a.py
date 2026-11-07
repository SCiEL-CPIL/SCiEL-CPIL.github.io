"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin

def _time_remaining():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior std at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Check deadline
    if _time_remaining() < 0.5:
        # Fallback: simple EI calculation
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite output
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
