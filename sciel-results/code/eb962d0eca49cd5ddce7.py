"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Clip sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term for exploration
    # Use a lower bound on mu for exploration: we want to explore where mu is low
    # but also where uncertainty is high
    ucb_explore = -mu + 0.7 * sigma_safe
    
    # Combine EI and exploration with adaptive weighting
    # Higher weight on EI when we have a good f_best, more exploration early
    # Use a soft blending: score = max(ei, alpha * ucb_explore) / cost
    # But we can also use a weighted sum
    
    # Let's try: score = (ei * 0.7 + ucb_explore * 0.3) / cost
    # This gives 70% weight to exploitation, 30% to exploration
    
    score = (0.7 * ei + 0.3 * ucb_explore) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
