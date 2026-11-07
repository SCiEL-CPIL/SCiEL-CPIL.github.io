"""Rewritten program: LCB-based cost-aware acquisition for 3-hump camel 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition normalized by cost.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better). Finite values only.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration coefficient:
    # We want to balance exploitation (low mu) and exploration (high sigma).
    # A common approach is to use a fixed beta, but we can make it adaptive.
    # For simplicity and robustness, we use a fixed beta that works well
    # for this problem scale. The objective values for 3-hump camel are small,
    # so beta around 1.0-2.0 is reasonable.
    # We can also normalize by the scale of the problem.
    # Let's use a fixed beta = 1.5 as a good starting point for LCB.
    beta = 1.5
    
    # LCB: we want to minimize mu - beta * sigma
    # So the acquisition score should be high when mu - beta * sigma is low.
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    lcb = mu - beta * sigma_safe
    scores = -lcb / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
