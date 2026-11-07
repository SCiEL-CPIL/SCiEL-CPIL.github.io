"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Quantile Acquisition Function (QAF) / Probability of Improvement (PI)
    # Maximizes the probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This directly optimizes for the chance of finding a new global minimum.
    
    # Avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) = norm.cdf(z)
    scores = norm.cdf(z)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
