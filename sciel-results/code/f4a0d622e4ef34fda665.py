"""Rewritten program using GP-LCB acquisition for cost-aware Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware GP-LCB acquisition function.
    
    Uses a Lower Confidence Bound (LCB) objective: mu - beta * sigma,
    then divides by cost to account for evaluation expense.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Beta controls exploration vs exploitation trade-off
    beta = 2.0
    
    # LCB: we want to minimize, so lower is better; LCB = mu - beta * sigma
    lcb = mu - beta * sigma
    
    # Cost-aware: divide by cost to get value per unit cost
    # Higher score means we should evaluate this point next
    # Since we're minimizing, we want low LCB values, so we negate
    # and divide by cost to get "improvement per cost"
    scores = -lcb / (cost + 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
