"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr, ndtri


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) normalized by cost.
    """
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit
    
    # Sanitize inputs to prevent numerical issues
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Calculate standardized improvement z
    z = (f_best - mu) / sigma
    
    # Clip z to a reasonable range to ensure numerical stability
    z = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement (EI) using stable special functions
    cdf_z = ndtr(z)
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # EI is theoretically non-negative, but numerical errors might make it slightly negative
    ei = np.maximum(ei, 0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 1.0:
        # If we're close to the deadline, return immediately
        return scores
    
    return scores
# EVOLVE-BLOCK-END
