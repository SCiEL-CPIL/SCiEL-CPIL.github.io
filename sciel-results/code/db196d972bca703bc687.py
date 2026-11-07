import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    LCB encourages exploration by penalizing high uncertainty (sigma).
    Score is normalized by cost to favor cheaper evaluations.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB for minimization: we prefer low mu and high sigma (exploration)
    # Value to maximize: f_best - mu + beta * sigma
    # Using beta = 1.0 as a standard starting point
    beta = 1.0
    lcb_values = f_best - mu + beta * sigma_safe
    
    # Cost-aware: divide by cost
    scores = lcb_values / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
