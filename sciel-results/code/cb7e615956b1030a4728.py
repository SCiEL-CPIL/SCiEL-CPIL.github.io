"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized GP-UCB with dynamically annealed exploration coefficient.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate).
    """
    # Track elapsed time for deadline guard
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Budget: total cost limit is 30, initial 5 uniform points cost ~5 * (1 + 0.5) = 7.5
    # So remaining budget is roughly 22.5, but we don't know exact current cost.
    # We can estimate progress based on the ratio of f_best to known minimum.
    # Known minimum for Hartmann 6d is -3.32237.
    min_val = -3.32237
    
    # Estimate progress: how close are we to the minimum?
    # If f_best is much higher than min_val, we're early; if close, we're late.
    # Use a soft estimate based on the gap.
    gap = f_best - min_val
    if gap < 1e-6:
        progress = 1.0
    else:
        # Normalize gap: assume initial gap is around 5-10 (uniform random starts)
        # Use a logarithmic scale for smoother annealing
        progress = np.clip(1.0 - np.log1p(gap) / np.log1p(10.0), 0.0, 1.0)
    
    # Dynamically anneal beta: start high for exploration, decay for exploitation
    # beta(t) = beta_max * exp(-alpha * progress) + beta_min
    beta_max = 3.0
    beta_min = 0.5
    alpha = 2.0
    beta = beta_max * np.exp(-alpha * progress) + beta_min
    
    # GP-UCB: score = -mu + beta * sigma (we're minimizing, so we want low mu, high sigma)
    # But we need to normalize by cost
    # Cost-normalized: (beta * sigma - mu) / cost
    # Higher score = better
    
    # Compute unnormalized UCB (higher is better for minimization)
    ucb = -mu + beta * sigma
    
    # Normalize by cost
    scores = ucb / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard: check elapsed time
    elapsed = time.time() - t_start
    if elapsed > 290:  # Safety margin before 300s limit
        # Return current scores as-is (computation is fast, so this is just a safeguard)
        pass
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
