import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Upper Confidence Bound (UCB) with a linear cost penalty.
    """
    # UCB for minimization:
    # We want to maximize the score.
    # Low mu (predicted minimum) is good.
    # High sigma (uncertainty) is good for exploration.
    # beta controls the exploration-exploitation trade-off.
    # In 6D with a small budget (~20 evaluations), exploration is critical.
    beta = 2.5
    
    # Calculate UCB score
    # mu and sigma are in objective units
    ucb = -mu + beta * sigma
    
    # Cost-aware scoring
    # Cost is 1 + mean(x), ranging from 1 to 2.
    # The true minimum's cost varies per instance (uniform in [1.1, 1.9]).
    # A linear penalty (exponent 1.0) is used to bias towards cheaper points
    # without excessively penalizing high-cost regions that might contain the true minimum.
    safe_cost = np.maximum(cost, 1e-9)
    score = ucb / safe_cost
    
    # Ensure all scores are finite to avoid invalidation
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
