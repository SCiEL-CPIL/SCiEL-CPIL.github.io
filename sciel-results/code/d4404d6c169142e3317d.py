"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill


def _is_time_exceeded():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted Thompson Sampling proxy.
    
    Args:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best (lowest) observed value
        cost: array of shape (2000,), cost of evaluating each point
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we are running too long, return a safe fallback
    if _is_time_exceeded():
        # Fallback: prioritize low cost and high uncertainty, but keep it simple
        return -cost + sigma * 0.1

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Sample lambda from a Beta distribution to stochastically shift
    # the exploration-exploitation trade-off per iteration
    # Using Beta(2, 3) which has mean 0.4, favoring exploitation slightly
    # but with enough exploration to escape local optima
    rng = np.random.default_rng()
    lam = rng.beta(2.0, 3.0)
    
    # Cost-Weighted Thompson Sampling proxy:
    # score = (lambda * (f_best - mu) + (1-lambda) * sigma) / cost
    # This balances exploitation (f_best - mu) and exploration (sigma)
    # with a random multiplier, then normalizes by cost
    
    exploitation = f_best - mu
    exploration = sigma
    
    # Combine with the sampled lambda
    combined = lam * exploitation + (1.0 - lam) * exploration
    
    # Normalize by cost to get cost-aware score
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
