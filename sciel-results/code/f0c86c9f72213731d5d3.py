# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    Uses Expected Improvement (EI) with robust numerical handling.
    
    Args:
        mu: array of shape (2000,), posterior mean at candidate points
        sigma: array of shape (2000,), posterior standard deviation at candidate points
        f_best: lowest observed value so far
    
    Returns:
        array of shape (2000,) with scores, higher meaning evaluate next
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robust sigma floor to avoid division by zero and extreme values
    # 1e-8 is a standard epsilon for this scale
    sigma_floor = 1e-8
    sigma_safe = np.maximum(sigma, sigma_floor)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to find points with low true values.
    # The optimistic lower bound for f(x) is mu - beta * sigma.
    # We maximize the score, so score = -(mu - beta * sigma) = beta * sigma - mu.
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Adaptive beta is not possible without n, so we use a fixed value.
    # 1.0 is a standard choice for Hartmann 3D with this number of evaluations.
    beta = 1.0
    
    score = beta * sigma_safe - mu
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
