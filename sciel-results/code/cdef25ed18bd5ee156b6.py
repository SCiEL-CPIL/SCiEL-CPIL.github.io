"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Budget-Decaying UCB.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (N,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (N,)
        Evaluation cost for each candidate point.
    
    Returns:
    --------
    np.ndarray, shape (N,)
        Acquisition values (higher = more promising).
    """
    global _call_count
    
    # Safety margin: return early if we're getting close to the 300s limit
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: simple exploitation
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-8)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.clip(cost, 1e-8, None)
    
    # Update call count to estimate budget usage
    _call_count += 1
    
    # Estimate total steps.
    # Initial 5 points cost ~7.5. Limit 30. Remaining ~22.5.
    # Avg cost ~1.5. Steps ~15.
    # Let's assume a max of 20 steps to be safe.
    total_steps_estimate = 20.0
    
    # Progress: 0.0 at start, 1.0 at end
    progress = min(1.0, _call_count / total_steps_estimate)
    
    # Decay beta from high (exploration) to low (exploitation)
    beta_start = 10.0
    beta_end = 0.5
    
    # Linear decay
    beta = beta_start - progress * (beta_start - beta_end)
    
    # UCB for minimization:
    # We want to maximize the score.
    # Low mu is good -> -mu is high.
    # High sigma is good (uncertainty) -> + beta * sigma is high.
    # High cost is bad -> divide by cost.
    
    # Score = (-mu + beta * sigma) / cost
    exploitation = -mu
    exploration = beta * sigma_safe
    raw_score = exploitation + exploration
    
    # Cost-aware normalization
    score = raw_score / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
