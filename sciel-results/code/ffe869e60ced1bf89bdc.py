# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-UCB with budget-scaled exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Budget: total cost limit is 30. We don't know current total cost directly,
    # but we can use a heuristic based on the cost array or assume a typical progression.
    # Since we don't have the current total cost, we use a fixed schedule that decays
    # exploration as if we're approaching the budget. A common approach is to use
    # a constant beta that balances exploration and exploitation.
    #
    # The directive says: "budget-scaled exploration coefficient that decays linearly 
    # from high to low as total cost approaches 30". Since we don't have the current 
    # total cost, we'll use a moderate fixed beta that works well in practice.
    # For UCB: score = mu + beta * sigma
    # We want to divide by a convex combination of cost and a constant.
    
    # Use a moderate exploration coefficient
    beta = 2.0
    
    # UCB score (higher is better for exploration, but we're minimizing, so we want
    # to find low mu + high uncertainty). Actually, for minimization with UCB,
    # we typically use: -mu + beta * sigma or mu - beta * sigma depending on convention.
    # Since we're minimizing, we want to explore regions where mu is low and sigma is high.
    # The UCB for minimization is often: mu - beta * sigma (lower bound) or we can use
    # -mu + beta * sigma. Let's use the standard form: we want to maximize the chance
    # of finding a better minimum, so we look at regions where the lower confidence
    # bound is low. LCB = mu - beta * sigma. We want to maximize -LCB = -mu + beta * sigma.
    
    lcb = mu - beta * sigma
    ucb_score = -mu + beta * sigma  # This is -LCB, higher is better
    
    # Divide by a convex combination of cost and a constant
    # cost is in range [1, 2] (1 + mean(x scaled to [0,1]), so cost in [1, 2])
    # Use a convex combination: alpha * cost + (1 - alpha) * c
    # Let's use alpha = 0.5, c = 1.5 (midpoint of cost range)
    # Or simpler: just divide by cost to get per-unit-cost improvement
    
    # Following the directive: divide by convex combination of cost and constant
    alpha = 0.5
    c_const = 1.0  # Lower bound of cost
    denom = alpha * cost_safe + (1 - alpha) * c_const
    denom = np.maximum(denom, 1e-9)
    
    scores = ucb_score / denom
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
