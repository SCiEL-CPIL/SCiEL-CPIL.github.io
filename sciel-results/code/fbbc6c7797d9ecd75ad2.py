"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with UCB exploration term
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # UCB: -mu + beta * sigma
    # Combined: EI + beta_ucb * sigma (where beta_ucb is the exploration coefficient)
    
    # Compute z for EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB exploration term (beta_ucb = 0.1 as suggested by inspiration)
    beta_ucb = 0.1
    ucb_term = beta_ucb * sigma
    
    # Combined acquisition: EI + UCB term
    scores = ei + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
