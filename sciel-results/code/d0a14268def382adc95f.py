# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: balance expected improvement against cost.
    # Soften the cost penalty so the search can reach the global minimum
    # which lies in an intermediate-cost region.
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by a softened cost term.
    # The cost ranges from 1 (cheap corner) to 2 (expensive corner).
    # Using a softer penalty (e.g., cost^0.5 or cost^0.7) allows the
    # algorithm to explore higher-cost regions that may contain the minimum.
    # Based on measured knowledge, softening from cost^1 to cost^~0.5 improved score.
    
    # Use a power less than 1 to soften the cost penalty
    cost_soft = np.power(cost, 0.5)
    
    # Also add a small exploration bonus for high uncertainty
    # This helps escape local minima
    uncertainty_bonus = sigma_safe * 0.1
    
    score = ei / cost_soft + uncertainty_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
