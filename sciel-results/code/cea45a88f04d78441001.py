"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost with clipping for stability.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    improvement = np.maximum(f_best - mu, 0.0)
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z_clipped = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost-aware: divide by cost
    scores = ei / np.maximum(cost, eps)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
