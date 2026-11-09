"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with a UCB-like exploration bonus,
    both normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Replace near-zero sigma with small value to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical issues: EI should be non-negative, clip negatives
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term: beta * sigma
    # This encourages exploration in regions with high uncertainty
    beta = 1.5
    ucb_bonus = beta * sigma_safe
    
    # Combine EI and UCB bonus
    # EI drives exploitation (finding improvements), UCB drives exploration
    combined = ei + ucb_bonus
    
    # Cost-aware: normalize by cost
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    scores = combined / cost_safe
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return best effort - we already have the result
        pass
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
