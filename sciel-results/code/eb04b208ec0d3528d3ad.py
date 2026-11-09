"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer to track elapsed time for the deadline guard
_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s/330s limits

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Thompson Sampling.
    
    Structural Change: Replaces deterministic EI/UCB blend with stochastic Thompson Sampling.
    This samples from the posterior GP and selects points based on the sampled value 
    normalized by cost, promoting better exploration in multimodal landscapes.
    """
    # Deadline Guard
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT:
        # Fallback to a simple, fast exploitation strategy to ensure termination
        # Return -mu / cost (greedy)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
        return -mu_safe / cost_safe

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Clip sigma
    sigma = np.clip(sigma, 1e-9, None)

    # Thompson Sampling: Sample from the posterior
    # y ~ N(mu, sigma^2)
    # We want to minimize f, so we want low y.
    # Cost-aware: We prefer low y relative to cost.
    # Score = -y / cost (higher is better)
    
    # Generate random normal samples
    z = np.random.normal(0, 1, size=mu.shape)
    y_sampled = mu + sigma * z
    
    # Compute score
    # To avoid numerical issues with division, we ensure cost is not zero
    score = -y_sampled / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
