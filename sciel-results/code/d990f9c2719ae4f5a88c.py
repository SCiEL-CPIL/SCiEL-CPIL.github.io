"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
TIME_LIMIT = 290  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI is a well-established acquisition function that balances exploration
    and exploitation by computing the expected improvement over the best
    observed value. It has strong theoretical properties and performs well
    on multimodal functions like Weierstrass.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Minimum observed value
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # If we are running out of time, fall back to a simple deterministic score
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    # Ensure numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle the case where sigma is very small
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF and PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
