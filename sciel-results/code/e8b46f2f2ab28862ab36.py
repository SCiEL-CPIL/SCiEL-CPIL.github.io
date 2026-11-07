"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by evaluation cost.
    
    This maximizes E[max(f_best - f(x), 0)] / cost(x).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        cost_safe = np.maximum(cost, 1e-8)
        # Fallback to simple LCB-like score
        fallback_scores = (sigma - mu) / cost_safe
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in exp/cdf
    z = np.clip(z, -10, 10)
    
    # Calculate PDF and CDF
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Expected Improvement:
    # If mu < f_best, the improvement is (f_best - mu) * P(f < f_best) + sigma * pdf(z)
    # Note: Standard EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware score: Higher is better
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
