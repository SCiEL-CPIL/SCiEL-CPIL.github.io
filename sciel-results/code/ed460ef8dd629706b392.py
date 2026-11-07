"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement per unit cost
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Score = PI(x) / cost(x)
    
    # Add small epsilon to avoid division by zero
    sigma_safe = sigma + 1e-9
    cost_safe = cost + 1e-9
    
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Ensure scores are finite
    scores = pi / cost_safe
    
    # Replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
