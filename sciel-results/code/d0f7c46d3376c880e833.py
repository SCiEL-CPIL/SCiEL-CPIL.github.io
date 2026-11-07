import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Powell 4D BO.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std predictions  
    - f_best: scalar, best observed value so far
    - cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    - (2000,) array of acquisition scores (higher = better)
    """
    # Time guard: if we've been running too long, return safe fallback
    elapsed = _get_elapsed()
    if elapsed > 280:  # Safety margin before 300s limit
        # Return simple EI/cost as fallback
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration term (UCB-like)
    # Adding a multiple of sigma encourages exploration in uncertain regions
    exploration_lambda = 1.0
    exploration_term = exploration_lambda * sigma
    
    # Combined exploitation and exploration
    base_score = ei + exploration_term
    
    # Structural Change: Exponential Cost Penalty
    # Cost is in [1, 2]. We penalize deviation from minimum cost (1.0).
    # cost_penalty = exp(-beta * (cost - 1.0))
    # This creates a smooth, non-linear penalty that is 1.0 at cost=1.0
    # and decays exponentially as cost increases.
    # beta controls the strength of the penalty.
    # beta=1.5 means cost=2.0 gets penalty exp(-1.5) ~ 0.22
    # beta=1.0 means cost=2.0 gets penalty exp(-1.0) ~ 0.37
    # beta=2.0 means cost=2.0 gets penalty exp(-2.0) ~ 0.13
    beta = 1.5
    cost_penalty = np.exp(-beta * (cost - 1.0))
    
    # Final score
    score = base_score * cost_penalty
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
