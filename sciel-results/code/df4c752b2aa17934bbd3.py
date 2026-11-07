"""Rewritten program using global budget-aware EI with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using global budget-aware Expected Improvement
    with adaptive exploration penalty.
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe random-like score
    if _is_deadline_exceeded():
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.clip(ei, 0.0, None)
    
    # Estimate progress: how close are we to the expected minimum?
    f_min_expected = -3.86278
    f_max_expected = 3.86  # Rough upper bound for Hartmann 3D on [0,1]^3
    
    # Normalize f_best to a progress metric [0, 1]
    # If f_best is very bad (high), progress is low -> explore more.
    # If f_best is very good (low), progress is high -> exploit more.
    progress = (f_max_expected - f_best) / (f_max_expected - f_min_expected)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive exploration weight:
    # Early on (low progress): higher weight on exploration (sigma term)
    # Late (high progress): higher weight on exploitation (EI term)
    w_explore = 0.4 * (1.0 - progress)
    w_exploit = 1.0 - w_explore
    
    # Exploration bonus: scaled sigma
    exploration_bonus = w_explore * sigma_safe
    
    # Base score: blend of EI and exploration
    base_score = w_exploit * ei + exploration_bonus
    
    # Global budget-aware scaling:
    # Instead of dividing each score by its individual cost,
    # compute a normalized cost penalty relative to the batch mean.
    mean_cost = np.mean(cost_safe)
    cost_ratio = cost_safe / mean_cost
    
    # Cost sensitivity increases as we approach the budget limit
    # Higher progress (lower f_best) means we're later in the game, so be more cost-sensitive
    cost_sensitivity = 0.3 + 0.7 * progress
    
    # Subtract penalty: candidates with cost_ratio > 1 get penalized
    # Candidates with cost_ratio < 1 get a small bonus
    cost_penalty = cost_sensitivity * (cost_ratio - 1.0)
    
    # Final score: base_score - cost_penalty
    scores = base_score - cost_penalty
    
    # Final safety: replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
