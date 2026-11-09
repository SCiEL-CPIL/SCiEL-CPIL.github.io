"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using pure GP-Lower Confidence Bound for minimization.
    
    score = -mu - kappa * sigma
    
    Higher scores indicate candidates with lower predicted objective values
    and/or higher uncertainty (more exploration potential).
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (unused in pure LCB)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    start = time.time()
    deadline = 325.0  # Leave 5 seconds margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and safe
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Fixed exploration parameter
    kappa = 2.0
    
    # GP-LCB for minimization: maximize -(mu + kappa*sigma) = -mu - kappa*sigma
    score = -mu - kappa * sigma_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    elapsed = time.time() - start
    if elapsed > deadline:
        return np.zeros_like(mu)
    
    return score
# EVOLVE-BLOCK-END
