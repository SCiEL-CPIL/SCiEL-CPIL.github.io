"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    For minimization, we maximize the score:
    score(x) = -mu(x) + beta * sigma(x)
    
    A high beta encourages exploration of uncertain regions, which is
    crucial for Branin 2D's multi-modal landscape to avoid getting stuck
    in local minima.
    """
    t_start = time.time()
    deadline = t_start + 295  # Safety margin
    
    n = len(mu)
    if n == 0:
        return np.array([])
        
    # Ensure inputs are numpy arrays and handle non-finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite mu with a large penalty (or median if available, but large penalty is safer for minimization)
    if not np.all(np.isfinite(mu)):
        finite_mu = mu[np.isfinite(mu)]
        if len(finite_mu) > 0:
            mu = np.where(np.isfinite(mu), mu, np.max(finite_mu) + 100)
        else:
            mu = np.zeros(n)
            
    # Ensure sigma is positive and finite
    if not np.all(np.isfinite(sigma)):
        finite_sigma = sigma[np.isfinite(sigma)]
        if len(finite_sigma) > 0:
            sigma = np.where(np.isfinite(sigma), sigma, np.median(finite_sigma))
        else:
            sigma = np.ones(n)
            
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB Acquisition
    # Beta = 5.0 as suggested for aggressive exploration
    beta = 5.0
    
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Time guard (unlikely to trigger for this simple calculation)
    if time.time() > deadline:
        return score
        
    return score
# EVOLVE-BLOCK-END
