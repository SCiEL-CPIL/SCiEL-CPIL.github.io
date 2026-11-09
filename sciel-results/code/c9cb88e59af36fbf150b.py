"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Thompson Sampling: sample y ~ N(mu, sigma^2) and score by -y / cost.
    This naturally balances exploration and exploitation without tuning hyperparameters.
    """
    # Deadline guard
    t_start = time.time()
    DEADLINE = 280.0  # Safety margin from 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: sample from predictive distribution
    # y ~ N(mu, sigma^2)
    rng = np.random.default_rng()
    y_sample = mu + sigma_safe * rng.standard_normal(n)
    
    # Score: lower y is better, higher cost is worse
    # We want to maximize the score, so we use -y / cost
    score = -y_sample / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - t_start > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
