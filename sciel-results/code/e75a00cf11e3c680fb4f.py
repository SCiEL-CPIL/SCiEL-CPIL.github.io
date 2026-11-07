"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill


def _is_time_exceeded():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Args:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best (lowest) observed value
        cost: array of shape (2000,), cost of evaluating each point
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if _is_time_exceeded():
        # Return a safe fallback: prioritize low cost, high uncertainty
        return -cost + sigma * 0.1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Clip mu to reasonable range to avoid numerical issues
    mu = np.clip(mu, f_best - 100.0, f_best + 100.0)
    
    # Compute z-score for Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid extreme values in CDF/PDF
    z = np.clip(z, -50.0, 50.0)
    
    # Expected Improvement (EI)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Use a soft cost penalty to avoid division issues
    cost_aware_ei = ei / cost
    
    # Add a small exploration bonus for high uncertainty relative to cost
    # This helps explore when EI is zero (e.g., mu > f_best)
    exploration_bonus = sigma / (cost * (1.0 + sigma))
    
    # Combine: primarily EI/cost, with small exploration term
    scores = cost_aware_ei + 0.05 * exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
