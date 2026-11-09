# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: Exploitation only (minimize mu, prefer low cost)
        return -mu / np.maximum(cost, 1e-9)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score
    Z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Expected Improvement
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Cost-aware score
    score = EI / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
