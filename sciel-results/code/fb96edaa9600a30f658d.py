"""Rewritten program: LCB-based cost-aware acquisition for 3-hump camel 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition with dynamic exploration coefficient,
    normalized by cost.
    
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
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic exploration coefficient based on f_best.
    # The 3-hump camel function has a minimum of 0 and a maximum around 4-5.
    # When f_best is high (poor), we want more exploration (higher beta).
    # When f_best is low (good), we want more exploitation (lower beta).
    # We map f_best from [0, 5] to beta [1.5, 4.0] to encourage more exploration
    # in the early stages and stronger exploitation as we converge.
    f_best_clipped = np.clip(f_best, 0.0, 5.0)
    beta = 1.5 + 2.5 * (f_best_clipped / 5.0)
    
    # LCB: we want to minimize mu - beta * sigma
    # So the acquisition score should be high when mu - beta * sigma is low.
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    lcb = mu - beta * sigma_safe
    scores = -lcb / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
