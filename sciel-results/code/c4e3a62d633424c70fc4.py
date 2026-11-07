import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with an exponential cost penalty.
    
    Strategy:
    1. Compute Probability of Improvement (PI): P(f(x) < f_best).
       This is robust to high-variance outliers that might inflate EI.
    2. Apply an exponential cost penalty: exp(-alpha * cost).
       This creates a non-linear, aggressive penalty for expensive evaluations,
       structurally different from the linear division used in EI/cost.
    3. Combine them multiplicatively.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: ensure sigma is positive to avoid division by zero in z-score
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score for improvement
    # z = (f_best - mu) / sigma
    # If mu is much lower than f_best, z is large positive, PI is ~1.
    # If mu is much higher, z is negative, PI is ~0.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    # Clip z to avoid numerical issues in cdf, though scipy is robust
    z_clipped = np.clip(z, -50, 50)
    pi = norm.cdf(z_clipped)
    
    # Cost Penalty
    # Cost is in range [1, 2] (1 + mean(x_scaled)).
    # We want to strongly prefer lower costs.
    # alpha controls the strength of the penalty.
    # If alpha is high, only very cheap points are considered.
    # If alpha is low, it behaves more like uniform or PI-only.
    # Given cost range is small [1, 2], a moderate alpha is needed.
    # exp(-alpha * cost)
    # Let's try alpha = 2.0. 
    # cost=1 -> exp(-2) ~ 0.135
    # cost=2 -> exp(-4) ~ 0.018
    # Ratio is ~7.5x preference for cheap over expensive.
    # With EI/cost, ratio is 2x. This is a much stronger structural shift.
    alpha = 2.0
    
    cost_penalty = np.exp(-alpha * cost)
    
    # Combined Score
    score = pi * cost_penalty
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If all scores are identical (e.g., very early on or flat GP), add small noise
    if np.max(score) - np.min(score) < 1e-15:
        score = score + np.random.randn(len(score)) * 1e-10
        
    return score
