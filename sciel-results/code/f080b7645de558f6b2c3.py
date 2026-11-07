"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    Includes a deadline guard to ensure robustness within the time limit.
    """
    start_time = time.time()
    
    # Safety check for sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Deadline Guard
    # The computation is vectorized and fast, but we check just in case.
    elapsed = time.time() - start_time
    if elapsed > 300:
        # Fallback: Return a simple UCB score to ensure we return *something*
        # This branch is highly unlikely to be taken for N=2000
        return -mu + sigma_safe

    return ei
# EVOLVE-BLOCK-END
