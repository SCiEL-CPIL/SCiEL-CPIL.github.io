# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of Expected Improvement (EI) and a penalty for high cost,
    with additional exploration via a UCB-like term to balance exploitation and exploration.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 325.0:  # Safety margin
        # Fallback: simple heuristic
        # Prefer low mu and low cost
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to a reasonable range to avoid numerical issues in cdf/pdf
    z_clipped = np.clip(z, -50.0, 50.0)
    
    # Use scipy's norm functions
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    # Compute EI
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI divided by cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Add a small UCB-like exploration term to encourage exploring high-uncertainty regions
    # This helps escape local optima in the acquisition landscape
    exploration_weight = 0.1
    ucb_term = exploration_weight * sigma_safe
    
    # Combined score
    score = (ei + ucb_term) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
