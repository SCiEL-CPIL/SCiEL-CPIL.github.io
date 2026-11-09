"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Hybrid Acquisition: EI + UCB term
    # The UCB term encourages exploration of areas with high uncertainty
    # Beta controls the trade-off. A value of 2.0 is a reasonable start.
    beta = 2.0
    ucb = beta * sigma_safe
    
    # Combined score
    score = ei + ucb
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Cost is guaranteed to be >= 1.0
    score = score / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
