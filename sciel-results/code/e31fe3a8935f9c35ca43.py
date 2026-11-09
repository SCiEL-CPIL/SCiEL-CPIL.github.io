"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer to enforce the 330s wall-clock limit with a safety margin
_start_time = time.time()
TIME_LIMIT = 320.0 # Leave 10s buffer for cleanup/return

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (CEI) acquisition function.
    Score = EI(x) / cost(x)
    """
    # Check time limit
    if time.time() - _start_time > TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    f_best_val = float(f_best)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best_val - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best_val - mu) * cdf_val + sigma_safe * pdf_val
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    scores = ei / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
