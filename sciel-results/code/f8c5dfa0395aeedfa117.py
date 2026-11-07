"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check time budget
    if time.time() - _start_time > _TIME_LIMIT:
        # Return simple UCB-like score as fallback
        return -mu + 1.0 * sigma
    
    # Expected Improvement with numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Use stable computation for EI
    # For large negative z, norm.cdf(z) ~ 0, norm.pdf(z) ~ 0
    # For large positive z, norm.cdf(z) ~ 1, norm.pdf(z) ~ 0
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add small exploration bonus for high uncertainty regions
    # This helps escape local optima in the RBF mixture
    exploration = 0.1 * sigma_safe
    
    return ei + exploration
# EVOLVE-BLOCK-END
