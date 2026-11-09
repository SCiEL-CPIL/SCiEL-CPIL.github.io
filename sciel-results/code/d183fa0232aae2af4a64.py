"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    This is a structural change from deterministic acquisition functions (like EI/UCB blends).
    Thompson Sampling selects candidates by sampling from the posterior GP and choosing
    the one that appears best (lowest value) relative to its cost.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x) where x is in [0,1]^3.
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    if _time_left() <= 0:
        # Fallback: simple cost-normalized negative mean
        return -mu / (cost + 1e-9)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Generate random normal variates for Thompson Sampling
    # Using a seed derived from the current best value and time to ensure
    # some variability if called multiple times in a tight loop, though
    # the evaluator calls it once per iteration.
    # Standard practice: just use global random state.
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample objective values from the posterior
    sampled_vals = mu + sigma_safe * z
    
    # Cost-aware score: We want to minimize sampled_vals / cost.
    # So we return negative of that ratio. Higher is better.
    scores = -sampled_vals / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
