"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    Uses mu + kappa * sigma divided by cost^gamma to balance exploration
    and exploitation while strongly favoring low-cost evaluations.
    """
    # Deadline guard: if we're close to the limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB: mu + kappa * sigma
    # Higher kappa means more exploration
    # We use a moderate kappa to balance exploration and exploitation
    kappa = 2.0
    
    ucb = mu + kappa * sigma_safe
    
    # Cost penalty: higher gamma means stronger preference for low-cost points
    # Cost is in [1, 2], so cost^3 ranges from 1 to 8
    gamma = 3.0
    
    # Cost-aware score: higher UCB and lower cost = better
    score = ucb / (cost ** gamma)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Negate because we want to minimize, but the evaluator picks the max score
    # Wait, the task says "higher = evaluate next" so we want high scores for good candidates
    # UCB is higher when mu is high or sigma is high
    # But we're minimizing, so we want LOW mu values
    # So we should use -mu + kappa * sigma
    
    # Let me reconsider: for minimization, we want low predicted values
    # So the acquisition should reward low mu and high sigma
    # score = -mu + kappa * sigma, then divide by cost^gamma
    
    # Recompute with correct sign
    ucb_min = -mu + kappa * sigma_safe
    
    score = ucb_min / (cost ** gamma)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
