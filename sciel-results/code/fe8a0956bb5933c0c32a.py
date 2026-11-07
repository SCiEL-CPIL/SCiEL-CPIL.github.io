"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI components
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (minimization perspective)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Fallback to simple EI/cost
        scores = ei / cost
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Revert to the seed's pure EI/cost strategy which scored 1.0.
    # The hybrid UCB approach in the previous version degraded performance.
    # Pure EI/cost is robust and well-calibrated for this task.
    scores = ei / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
