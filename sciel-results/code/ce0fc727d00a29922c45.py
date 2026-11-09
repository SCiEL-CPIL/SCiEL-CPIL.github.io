import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize the objective, so we look for low mu and high sigma (uncertainty)
    # Score to maximize: -mu + kappa * sigma
    
    # kappa controls the exploration-exploitation tradeoff
    # A value of 2.0 is a standard choice for BO
    kappa = 2.0
    
    # Calculate the raw UCB score
    # Higher score indicates a more promising candidate
    raw_score = -mu + kappa * sigma
    
    # Cost-aware: Divide by cost
    # Ensure cost is not zero
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)
    
    scores = raw_score / cost_safe
    
    # Handle any numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
