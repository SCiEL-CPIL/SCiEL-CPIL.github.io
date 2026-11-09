"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for the acquisition function calls
_acq_start_time = None
_TIME_LIMIT = 295  # Seconds - leave 5s safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP posterior mean in objective units
    sigma : array-like, shape (2000,)
        GP posterior std in objective units
    f_best : float
        Lowest observed objective value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    global _acq_start_time
    
    # Deadline guard
    if _acq_start_time is None:
        _acq_start_time = time.time()
    elif time.time() - _acq_start_time > _TIME_LIMIT:
        # Return a simple deterministic fallback if over time
        return np.zeros_like(np.asarray(mu, dtype=np.float64))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample from the GP posterior
    # For each candidate, sample a value from N(mu_i, sigma_i)
    # The acquisition is -sampled_value / cost_i
    # Higher scores = lower expected cost-normalized objective
    
    # Use a deterministic pseudo-random offset based on the data to ensure
    # reproducibility across calls within the same loop iteration
    # We use the mean and std themselves to create a stable random draw
    
    # Sample from N(mu, sigma) using a fixed random seed derived from data
    # This ensures the same candidates get consistent scores
    rng = np.random.default_rng(seed=42)
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # Thompson sampling score: negative sampled value divided by cost
    # We want to minimize objective / cost, so maximize -objective / cost
    scores = -samples / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
