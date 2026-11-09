"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Discrete, cost-normalized quantile-based heuristic for acquisition.
    
    Ranks candidates by the ratio of their posterior probability of exceeding
    the current best to their cost, decoupling from noisy mean estimates and
    prioritizing low-cost regions with high relative improvement likelihood.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = 320  # safety margin from 330s limit
    
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Cost-Aware UCB (Upper Confidence Bound)
    # We want to minimize f(x).
    # UCB for minimization: minimize (mu - kappa * sigma)
    # Acquisition (to maximize): kappa * sigma - mu
    # Normalize by cost to favor cheap evaluations.
    
    # Choose exploration parameter kappa
    # A value of 1.0 is a standard starting point.
    kappa = 1.0
    
    # Calculate UCB score
    ucb_score = kappa * sigma_safe - mu
    
    # Cost normalization
    cost_weight = 1.0 / np.maximum(cost, 1e-10)
    
    # Final score
    score = ucb_score * cost_weight
    
    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
