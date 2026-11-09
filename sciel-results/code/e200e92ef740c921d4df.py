"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Lower Confidence Bound (LCB) for minimization with adaptive beta.
    The score is -mu + beta * sigma, where beta adapts to the posterior
    uncertainty structure to balance exploration and exploitation.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e8, neginf=1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e8, neginf=-1e8)
    
    # Adaptive beta: use a value that scales with the ratio of 
    # exploration potential to exploitation potential.
    # When uncertainty is high relative to the gap between mu and f_best,
    # we explore more.
    
    # Compute the "exploitation pressure": how close are we to f_best?
    # If mu is already below f_best, exploitation is strong.
    # Use a soft measure based on the distribution of (f_best - mu)
    
    # Standard deviation of the posterior mean spread
    mu_spread = np.std(mu) if np.std(mu) > 1e-10 else 1e-8
    
    # Adaptive beta: 
    # - Start with a base value
    # - Scale by how much uncertainty there is relative to the function spread
    # - Cap to avoid pure exploration
    base_beta = 2.0
    uncertainty_ratio = np.median(sigma) / mu_spread
    beta = base_beta * np.clip(uncertainty_ratio, 0.5, 3.0)
    
    # LCB for minimization: we want to minimize mu - beta*sigma
    # Score = -mu + beta * sigma (higher is better)
    scores = -mu + beta * sigma
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e8, neginf=-1e8)
    
    return scores
# EVOLVE-BLOCK-END
