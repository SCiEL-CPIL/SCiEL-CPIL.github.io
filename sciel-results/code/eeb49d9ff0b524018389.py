"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) acquisition function.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the "remaining budget fraction" proxy.
    # The total cost budget is 30. The current spent cost is not directly given,
    # but we can infer a proxy from the cost array itself or use a fixed heuristic.
    # Actually, the problem states "While total cost is below 30". 
    # We don't have the current total cost passed in. 
    # However, we can estimate the urgency based on the cost of the candidates themselves.
    # A better approach for LCB in cost-aware BO is to use a beta parameter.
    # Let's use a standard LCB with a cost penalty.
    
    # Standard LCB: mu - beta * sigma
    # Cost-adjusted: (mu - beta * sigma) / cost
    # Or: mu - beta * sigma - lambda * cost
    
    # The directive suggests: "scales the uncertainty penalty by the inverse of the remaining budget fraction"
    # Since we don't have the exact remaining budget, let's use a robust heuristic.
    # Typically, as we spend more, we should exploit more.
    # Without the exact spent cost, we can use the mean cost of the candidates as a proxy for the "current" cost level,
    # or simply use a fixed beta that balances exploration and exploitation well for this problem.
    
    # Let's try a simple cost-normalized LCB first, which is a common strong baseline.
    # Score = (mu - beta * sigma) / cost
    
    # Choosing beta: 
    # For 4D Shekel, a beta around 2-5 often works well.
    # Let's use beta = 3.0 as a starting point.
    
    beta = 3.0
    
    # Compute LCB scores
    lcb_scores = mu - beta * sigma_safe
    
    # Normalize by cost to prefer cheaper points
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = lcb_scores / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
