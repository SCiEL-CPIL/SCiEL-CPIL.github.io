"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 330s wall-clock, return safely before limit.
_DEADLINE = time.time() + 325.0  # 5s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition score.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Higher = more desirable to evaluate next.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a deterministic fallback: prefer low cost
        return -cost.copy()
    
    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample one y ~ N(mu, sigma) per candidate
    # Use a fixed seed per call? No – the loop calls this repeatedly.
    # We just draw fresh samples each time.
    rng = np.random.default_rng()
    y = mu + sigma * rng.standard_normal(len(mu))
    
    # Score: improvement over f_best per unit cost
    # Higher is better: we want to minimize, so (f_best - y) is positive when y < f_best
    scores = (f_best - y) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
