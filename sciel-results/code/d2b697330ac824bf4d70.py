"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # 295 seconds limit (5s safety margin before 300s kill)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining dynamic LCB with additive cost penalty.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return 1/cost to prefer cheap points
        return 1.0 / (np.maximum(cost, 1e-9))
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic beta based on uncertainty budget
    # As we find better solutions (lower f_best), we can afford to be more exploitative
    # When f_best is high (early on), we need more exploration
    # Scale beta inversely with f_best magnitude, clamped to reasonable range
    # f_best starts high (rastrigin can be ~40-50) and decreases toward 0
    # beta should start high (exploration) and decrease (exploitation)
    f_best_abs = abs(f_best) + 1e-9
    # Map f_best from [0, 50] to beta [1.0, 3.5]
    # beta = 3.5 - 2.5 * (f_best_abs / 50.0)
    # Clamp to [1.0, 3.5]
    beta = 3.5 - 2.5 * min(f_best_abs / 50.0, 1.0)
    beta = np.clip(beta, 1.0, 3.5)
    
    # Calculate the optimistic gap (LCB-based)
    # gap = f_best - mu + beta * sigma
    gap = f_best - mu + beta * sigma_safe
    
    # Structural change from inspiration 1: Use additive cost penalty
    # This allows high-improvement points to overcome high costs more linearly
    # than division, which dampens high improvements
    # Score = gap - lambda * cost
    # 
    # Lambda should be tuned so that the cost penalty is meaningful
    # Cost is in [1, 2], gap can range from negative to large positive
    # Start with lambda = 1.0 and see if it helps
    lambda_cost = 1.0
    
    scores = gap - lambda_cost * cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
