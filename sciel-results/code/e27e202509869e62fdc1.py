"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Lower bound of Expected Improvement (LBEI) with a small exploration bonus.
    Uses the lower bound of EI which is more robust to noise and avoids
    over-exploration of regions with high uncertainty but low potential.
    """
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-8)
    
    # Standardized distance from the best
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of the standard normal distribution.
    
    # Calculate terms
    term1 = (f_best - mu) * norm.cdf(z)
    term2 = sigma * norm.pdf(z)
    
    scores = term1 + term2
    
    # Ensure non-negative scores (numerical stability)
    scores = np.maximum(scores, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
