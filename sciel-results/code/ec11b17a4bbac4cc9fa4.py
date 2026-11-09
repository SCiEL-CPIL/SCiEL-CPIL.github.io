"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lagrangian relaxation with linear score.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Calculate potential improvement (linear surrogate)
    # We are minimizing, so improvement is f_best - mu
    improvements = f_best - mu
    
    # Dynamically tune lambda
    # We want the penalty lambda * cost to be comparable to the typical improvement.
    # Use the mean of positive improvements as a scale factor.
    # This ensures that if most points are worse than f_best, lambda is small (exploit).
    # If many points are better, lambda is large (penalize cost more to balance).
    
    positive_imps = np.maximum(improvements, 0.0)
    scale = np.mean(positive_imps)
    
    mean_cost = np.mean(cost)
    
    if mean_cost > 0 and scale > 0:
        # Tuning factor: 1.0 means we balance mean improvement with mean cost.
        # We can adjust this alpha if needed, but 1.0 is a standard start.
        alpha = 1.0
        lambda_val = (scale / mean_cost) * alpha
    else:
        lambda_val = 0.0
        
    # Ensure lambda is non-negative
    lambda_val = max(lambda_val, 0.0)
    
    # Compute scores: f_best - mu - lambda * cost
    scores = improvements - lambda_val * cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
