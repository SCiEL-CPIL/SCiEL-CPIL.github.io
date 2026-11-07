# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses an adaptive Cost-Weighted Lower Confidence Bound (CW-LCB).
    
    Parameters:
    - mu: mean of GP predictions, shape (2000,)
    - sigma: std dev of GP predictions, shape (2000,)
    - f_best: best observed function value
    - cost: cost of evaluating each candidate, shape (2000,)
    
    Returns:
    - scores: acquisition values, shape (2000,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Deadline guard: if we are running out of time, return a simple heuristic
    # to ensure we don't get killed. However, the computation below is O(N) and very fast.
    # We skip complex checks for now as N=2000 is small.
    
    # Adaptive parameters based on f_best
    # f_best is the current best (lowest) value.
    # As f_best decreases (approaches 0), we become more sensitive to cost.
    
    # Exploration parameter beta:
    # Higher beta encourages exploring high-uncertainty regions.
    # We start with higher exploration and reduce it as we converge.
    # exp(-f_best / scale) is 1 when f_best=0, and decays as f_best increases.
    # We want MORE exploration when f_best is HIGH (far from optimum) to find the basin.
    # So we use exp(-f_best) which is small when f_best is large? No.
    # If f_best is large (e.g. 3), we are far. We want to explore.
    # If f_best is small (e.g. 0.1), we are close. We want to exploit.
    # So beta should be large when f_best is large.
    # Let's use: beta = beta_min + beta_max * (1 - exp(-f_best / 1.0))
    # If f_best = 0, beta = beta_min.
    # If f_best = 3, beta = beta_min + beta_max * (1 - 0.05) ~ beta_min + beta_max.
    
    beta_min = 0.5
    beta_max = 1.5
    beta = beta_min + beta_max * (1.0 - np.exp(-f_best / 1.0))
    
    # Cost sensitivity parameter lambda:
    # We want to penalize cost more when f_best is small (close to optimum).
    # lambda should be large when f_best is small.
    # lambda = lambda_max / (1 + f_best)
    # If f_best = 0, lambda = lambda_max.
    # If f_best = 3, lambda = lambda_max / 4.
    
    lambda_max = 0.8
    lambda_cost = lambda_max / (1.0 + f_best)
    
    # Compute the Cost-Weighted LCB score
    # We want to MINIMIZE mu - beta*sigma + lambda*cost
    # So we MAXIMIZE -mu + beta*sigma - lambda*cost
    
    scores = -mu + beta * sigma_safe - lambda_cost * cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf (should not happen), fallback to random
    if np.all(scores == -np.inf):
        scores = np.random.rand(len(mu))
        
    return scores
# EVOLVE-BLOCK-END
