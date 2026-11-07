"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with a UCB-style exploration term,
    normalized by evaluation cost. Includes a deadline guard for safety.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB-like exploration term: encourages visiting uncertain regions
    # Use a moderate beta to balance exploration
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB: weighted sum
    # EI is in objective units (higher is better for improvement)
    # UCB is in objective units (higher means more promising/uncertain)
    # Scale UCB to be comparable to EI
    # Use a blend: primarily EI with some UCB
    score = 0.7 * ei + 0.3 * ucb
    
    # Normalize by cost (cost-aware)
    cost_safe = np.maximum(cost, 1e-9)
    score = score / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    if time.time() - t0 > 280:
        # Return a simple fallback if taking too long
        return ei / cost_safe
    
    return score
# EVOLVE-BLOCK-END
