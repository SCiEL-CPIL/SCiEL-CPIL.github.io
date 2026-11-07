# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Effective UCB.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed value so far
    cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Exploration parameter for UCB
    beta = 1.0
    
    # Safe division
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # UCB for minimization: L(x) = mu(x) - beta * sigma(x)
    # We want to minimize L(x).
    # Score is higher for lower L(x) and lower cost.
    # Score = -L(x) / cost(x) = (beta * sigma(x) - mu(x)) / cost(x)
    scores = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
