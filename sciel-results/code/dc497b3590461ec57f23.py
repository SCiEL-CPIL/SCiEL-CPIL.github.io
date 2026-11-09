import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value (scalar)
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        scores: array of acquisition scores, shape (2000,)
    """
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, 10s safety margin

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Handle cases where sigma is very small to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement (PI) for minimization
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    # Normalized improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute PI
    # Clip z to avoid extreme values that cause numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    pi = norm.cdf(z_clipped)
    
    # Normalize PI by cost to the power of 0.5
    # This balances exploration efficiency against computational budget
    pi_score = pi / np.sqrt(cost_safe)
    
    # Ensure finite scores
    scores = np.nan_to_num(pi_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
