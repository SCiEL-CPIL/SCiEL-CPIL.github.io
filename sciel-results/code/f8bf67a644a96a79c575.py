"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with exploration bonus.
    # Add a small exploration term to avoid getting stuck in local minima
    # and to encourage exploration when uncertainty is high.
    sigma_safe = np.maximum(sigma, 1e-6)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage points with high uncertainty
    # Use a small coefficient to balance exploitation and exploration
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration, then normalize by cost
    score = (ei + exploration_bonus) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
