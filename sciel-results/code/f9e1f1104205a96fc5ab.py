"""Direct cost-aware acquisition for Bayesian optimization on griewank 2d."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: Cost-Adjusted Lower Confidence Bound (LCB).
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions in objective units
    - sigma: array of shape (2000,) - GP std-dev predictions in objective units
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs (1 + mean of scaled coords)
    
    Returns:
    - scores: array of shape (2000,) of finite scores, higher = more promising
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division/log issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Bayesian model evidence criterion:
    # score = -(f_best - mu) / sigma - log(sigma)
    # This approximates the expected negative log-likelihood of observing f_best
    # under the GP predictive distribution N(mu, sigma^2).
    # It naturally balances exploitation (low mu) against exploration (high sigma)
    # without arbitrary hyperparameters.
    
    # Compute the evidence-based score
    # Higher mu (closer to f_best) and lower sigma give higher scores
    scores = (mu - f_best) / sigma_safe - np.log(sigma_safe)
    
    # Weight by inverse cost to account for evaluation costs
    scores = scores / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
