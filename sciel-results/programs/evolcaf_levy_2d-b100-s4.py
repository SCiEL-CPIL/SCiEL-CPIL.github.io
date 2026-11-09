"""Cost-aware acquisition function for Bayesian optimization of Levy 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement with adaptive cost penalty.
    Uses a combination of exponential decay and inverse cost to balance
    exploitation of low-cost high-EI regions against exploration.
    """
    # Ensure proper types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost penalty:
    # 1. Inverse cost (linear penalty for expensive points)
    # 2. Exponential decay (stronger suppression of very expensive points)
    # The cost is in [1, 2] range. We use a combination:
    # - Base: 1/cost (linear)
    # - Exponential: exp(-alpha * (cost - 1)) to suppress high-cost regions
    #   where (cost - 1) is in [0, 1]
    
    # Normalize cost to [0, 1] for the exponential term
    cost_norm = (cost_safe - 1.0)  # Now in [0, 1]
    
    # Exponential penalty with moderate strength
    alpha = 2.0
    exp_penalty = np.exp(-alpha * cost_norm)
    
    # Combined score: EI * (1/cost) * exp_penalty
    # This gives both linear and exponential suppression of expensive regions
    scores = ei * (1.0 / cost_safe) * exp_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
