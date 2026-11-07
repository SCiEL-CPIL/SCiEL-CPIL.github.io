"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Return simple scores to avoid timeout
        return -mu
    
    # Expected Improvement with exploration parameter
    # Use a small exploration parameter to balance exploitation/exploration
    xi = 0.01 * (f_best - np.min(mu))
    
    # Clip sigma to avoid division issues
    sigma_clipped = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu - xi) / sigma_clipped
    
    # Expected improvement
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
