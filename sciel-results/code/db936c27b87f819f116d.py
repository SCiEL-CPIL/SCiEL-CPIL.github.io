"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with a gentler cost normalization (cost^0.5) to balance exploration/exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Add small epsilon to avoid division by zero
    sigma_safe = sigma + 1e-9
    
    # Probability of Improvement (PI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # PI = P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Use a gentler cost penalty: divide by cost^0.5 instead of cost
    # This allows better exploration of low-cost regions early
    # while still favoring exploitation later
    cost_factor = np.sqrt(np.maximum(cost, 1e-9))
    
    # Score is PI per unit sqrt(cost)
    scores = pi / cost_factor
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
