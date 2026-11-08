"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of:
    - Probability of Improvement (POI) for exploration
    - Expected Improvement (EI) for exploitation
    - Cost normalization for efficiency
    
    The POI component helps avoid premature convergence in high-dimensional
    multimodal landscapes, while cost-awareness ensures efficient budget use.
    """
    # Time guard: track elapsed time
    t_start = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Combine EI and POI with weighting
    # POI is more conservative and helps with exploration
    # EI drives exploitation
    # Use a blend that emphasizes POI for exploration
    combined = 0.3 * poi + 0.7 * ei
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # But add a small constant to avoid division by very small costs
    score = combined / (cost + 0.1)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time check: if we're taking too long, return early with current result
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:  # Safety margin
        return score
    
    return score
# EVOLVE-BLOCK-END
