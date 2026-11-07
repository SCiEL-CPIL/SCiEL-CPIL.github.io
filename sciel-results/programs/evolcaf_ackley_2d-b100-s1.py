"""Rewritten program using GP-LCB acquisition for cost-aware Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware GP-LCB acquisition function with tuned exploration and cost penalty.
    
    Uses a Lower Confidence Bound (LCB) objective: mu - beta * sigma.
    Applies a softer cost penalty (1/sqrt(cost)) to balance exploration of expensive regions.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Reduced beta to favor exploitation once a good region is found,
    # preventing excessive exploration of high-variance (often expensive) areas.
    beta = 1.0
    
    # LCB: we want to minimize, so lower is better; LCB = mu - beta * sigma
    lcb = mu - beta * sigma
    
    # Cost-aware: Use a softer penalty 1/sqrt(cost) instead of 1/cost.
    # This allows the algorithm to consider expensive points if the GP mean is sufficiently low,
    # which is crucial for finding the global minimum if it lies in a high-cost region.
    cost_penalty = np.sqrt(cost + 1e-9)
    scores = -lcb / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
