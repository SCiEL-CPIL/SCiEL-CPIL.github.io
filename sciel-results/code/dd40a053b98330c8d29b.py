import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    
    # Ensure sigma is not zero to avoid division issues in any intermediate steps
    # Although UCB doesn't divide by sigma, EI does. We use UCB.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score, which corresponds to minimizing the optimistic value.
    # Optimistic Value = mu - beta * sigma
    # Score = -Optimistic Value = -mu + beta * sigma
    beta = 2.0
    
    ucb_scores = -mu + beta * sigma_safe
    
    # Cost-aware scaling
    # Divide by cost to penalize expensive evaluations
    cost_safe = np.maximum(cost, 1e-9)
    scores = ucb_scores / cost_safe
    
    # Check time
    if time.time() - start_time > 290.0:
        # If we are running out of time, we still return the scores calculated so far.
        # The calculation is O(N) and very fast, so this is unlikely.
        pass
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
