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
    Cost-aware Expected Improvement (CEI) acquisition function with adaptive exploration.
    Score = EI(x) / cost(x)
    Includes a small bonus for points where GP predicts much better than f_best.
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
    
    # Add a small bonus for points where GP predicts much better than f_best
    # This encourages exploitation of very promising regions
    # Bonus is proportional to how much better mu is than f_best, scaled by confidence
    # confidence is high when sigma is small
    mu_better = np.maximum(f_best_val - mu, 0.0)
    confidence = 1.0 / (1.0 + sigma_safe)  # High when sigma is small
    bonus = 0.5 * mu_better * confidence
    
    # Combine EI and bonus
    scores_raw = ei + bonus
    
    # Cost-aware score
    scores = scores_raw / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
