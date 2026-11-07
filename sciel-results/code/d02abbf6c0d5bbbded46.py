import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Upper Confidence Limit (UCL) for minimization.
    
    Args:
        mu: Array of GP means, shape (N,)
        sigma: Array of GP standard deviations, shape (N,)
        f_best: Current best function value (float)
        cost: Array of evaluation costs, shape (N,)
        
    Returns:
        Array of scores, shape (N,). Higher score = more likely to be evaluated.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 320.0  # 10s safety margin from 330s limit
    
    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # UCL (Upper Confidence Limit) for minimization
    # We want to find points with low (mu - beta * sigma)
    # Score is negative of this, normalized by cost
    beta = 2.0
    ucl_values = mu - beta * sigma
    scores = -ucl_values / cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (unlikely to be triggered for vectorized operations)
    if time.time() - start_time > deadline:
        # If we hit the deadline, we just return what we have
        # In a more complex iterative acquisition, we would stop early
        pass
        
    return scores
