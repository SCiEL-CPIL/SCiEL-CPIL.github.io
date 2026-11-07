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
    Cost-aware acquisition function combining Expected Improvement and Upper Confidence Bound.
    
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
    
    # Cost-aware Lower Confidence Bound (LCB)
    # We want to minimize f(x).
    # LCB = mu - beta * sigma
    # We want to maximize the "optimistic gap" relative to f_best.
    # Score = (f_best - (mu - beta * sigma)) / cost
    #       = (f_best - mu + beta * sigma) / cost
    
    # Beta balances exploration (sigma) and exploitation (mu).
    # In multimodal Rastrigin, we need sufficient exploration to escape local minima.
    beta = 2.5
    
    # Calculate the numerator: optimistic potential improvement
    # If mu is much lower than f_best, this is large positive.
    # If sigma is high, this adds an exploration bonus.
    numerator = f_best - mu + beta * sigma_safe
    
    # Divide by cost to prefer cheaper evaluations
    # Cost is 1 + mean(x_scaled), so cheaper points (near center) get higher score
    cost_safe = np.maximum(cost, 1e-9)
    scores = numerator / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Clip negative scores to 0? 
    # If all scores are negative, we pick the max (least negative).
    # Clipping to 0 might make all "bad" points equal, leading to random choice among them.
    # It is better to keep the relative ordering of negative scores.
    # However, if we want to strictly prefer points that *could* be better, we might clip.
    # Let's keep the raw scores to maintain gradient information even in bad regions.
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
