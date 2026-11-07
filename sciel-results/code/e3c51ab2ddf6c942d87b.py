"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    EI provides a continuous score that balances exploitation (low mu) 
    with exploration (high sigma), unlike POI which is binary-like.
    
    Args:
        mu: Posterior mean, shape (N,)
        sigma: Posterior standard deviation, shape (N,)
        f_best: Best observed value so far.
        
    Returns:
        Scores, shape (N,). Higher is better.
    """
    # Ensure sigma is non-negative to avoid division by zero or negative sqrt issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    scores = (f_best - mu) * Phi + sigma * phi
    
    # Ensure scores are finite and non-negative
    scores = np.maximum(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
