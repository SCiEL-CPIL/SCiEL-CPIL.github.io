"""Cost-aware LCB acquisition for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition normalized by cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points (objective units).
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Dynamic exploration coefficient:
    # Start with higher exploration, decay as we get closer to the optimum.
    # The gap between f_best and the known minimum (0) indicates how much room is left.
    # Use a simple heuristic: beta scales with the remaining gap.
    gap = max(f_best - 0.0, 0.0)
    # Beta ranges from about 2.0 (large gap, need exploration) to about 0.5 (small gap, exploit)
    beta = 0.5 + 1.5 * min(gap / 10.0, 1.0)
    
    # LCB: we want to minimize mu - beta * sigma
    # As an acquisition score (higher is better), we use: -(mu - beta * sigma) = -mu + beta * sigma
    lcb_score = -mu + beta * sigma
    
    # Normalize by cost to get cost-aware score
    scores = lcb_score / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
