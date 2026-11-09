"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: Blend of Expected Improvement (EI) and UCB-like exploration.
    # EI exploits known good regions.
    # The exploration term encourages visiting uncertain regions.
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB Exploration term for minimization:
    # We want to maximize the score.
    # High sigma is good. Low mu is good (handled by EI mostly, but UCB adds to it).
    # A common blend is EI + beta * sigma.
    # Tuning beta: For 1D functions with few evaluations, moderate exploration is key.
    # Let's use beta = 2.0. This effectively adds 2 standard deviations of uncertainty 
    # to the expected improvement, pushing the search towards high-variance areas 
    # which is crucial for escaping local optima in Styblinski-Tang.
    
    beta = 2.0
    score = ei + beta * sigma_safe
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
