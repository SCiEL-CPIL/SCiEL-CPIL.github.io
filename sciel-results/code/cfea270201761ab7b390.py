"""Combined program: Cost-aware acquisition with deadline guard and adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 325.0  # Safety margin of 5s before the 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an adaptive exploration bonus, guarded by a deadline check.
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    # Deadline guard: If we are close to the limit, return a fast, conservative score
    # to ensure we finish safely. -mu/cost prioritizes exploitation of the current
    # best region while respecting cost.
    if time.time() > _DEADLINE - 2.0:
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-9, cost, 1.0)

    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # For minimization, improvement is f_best - f(x).
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # 2. Add an exploration bonus (UCB-like term)
    # In high dimensions (8d) with few samples, pure EI can be too greedy.
    # Adding a beta * sigma term encourages exploring high-uncertainty regions.
    # We use a modest beta to balance exploitation and exploration.
    beta = 1.5
    exploration_bonus = beta * sigma
    
    # 3. Combine EI and exploration
    # We weight them such that EI is the primary driver but exploration helps escape local minima.
    # A simple sum is often effective.
    combined_score = ei + exploration_bonus
    
    # 4. Cost-awareness: Divide by cost
    # Prioritize cheaper evaluations. Cost is in [1, 2].
    scores = combined_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
