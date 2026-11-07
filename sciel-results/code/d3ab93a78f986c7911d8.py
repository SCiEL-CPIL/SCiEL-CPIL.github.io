import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound
    # We want to minimize the function, so we maximize (kappa * sigma - mu) / cost
    # kappa controls the exploration-exploitation trade-off
    
    kappa = 2.0
    
    # Calculate UCB score
    # Note: For minimization, UCB is typically mu - kappa*sigma (lower bound).
    # We want to maximize the potential for a low value.
    # So we maximize -(mu - kappa*sigma) = kappa*sigma - mu.
    
    ucb_score = kappa * sigma - mu
    
    # Cost-aware: divide by cost
    # Cost is > 0
    score = ucb_score / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
