"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for the deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 325 seconds to be safe against the 330s kill switch

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses Expected Improvement (EI) as the primary acquisition function,
    combined with a small UCB term to ensure exploration in regions
    where EI is zero or very small.
    
    Parameters:
    mu: numpy array of shape (N,), posterior mean
    sigma: numpy array of shape (N,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    numpy array of shape (N,), acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a deterministic fallback: pure exploitation (lowest mean)
        return -mu
    
    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Use norm.cdf and norm.pdf
    # Note: for large negative z, cdf is 0, pdf is 0 -> EI is 0
    # For large positive z, cdf is 1, pdf is 0 -> EI is f_best - mu
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # 2. UCB component for minimization
    # We want to maximize the score, so we want low mu and high sigma.
    # Score_ucb = -mu + beta * sigma
    # A small beta ensures this term doesn't dominate EI, but helps break ties
    # and explore slightly uncertain regions.
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # 3. Combine EI and UCB
    # We add a small weight to UCB to ensure exploration.
    # The weight is small so that EI dominates when there is clear improvement potential.
    # If EI is 0 everywhere, UCB will drive the selection.
    weight_ucb = 0.01
    
    score = ei + weight_ucb * ucb
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
