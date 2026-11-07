import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    t_start = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Cost-aware UCB for minimization
    # Score = (Exploration - Exploitation) / Cost
    # Exploration: beta * sigma
    # Exploitation: -mu (since we want low mu)
    
    beta = 1.5
    
    # Calculate raw UCB score
    # Using a small epsilon to avoid division by zero if cost is 0 (though cost >= 1)
    safe_cost = np.maximum(cost, 1e-6)
    
    score = (beta * sigma - mu) / safe_cost
    
    # Check for non-finite values
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, 0.0)
        
    # Deadline Guard
    # If we are running too close to the limit, return a simple greedy score
    # This is unlikely to trigger for such a small array, but required by directive
    if time.time() - t_start > 295.0:
        greedy_score = -mu / safe_cost
        if not np.all(np.isfinite(greedy_score)):
            greedy_score = np.where(np.isfinite(greedy_score), greedy_score, 0.0)
        return greedy_score
        
    return score
