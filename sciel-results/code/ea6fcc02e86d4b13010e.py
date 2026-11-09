# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and bail out early with a safe fallback.
_DEADLINE = 325.0  # seconds from process start (safety margin)
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Ackley 2D.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Return a safe fallback: uniform random scores
        return np.random.uniform(0, 1, size=mu.shape)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # EI calculation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative EI
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # cost is already given, ensure it's positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
