import numpy as np
import time

# Deadline guard: track start time to ensure we finish before 330s
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for minimization.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: Scalar, the best objective value observed so far.
        cost: (2000,) array of evaluation costs (1 + mean(x)).
        
    Returns:
        (2000,) array of acquisition scores.
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    # Convert to numpy arrays for consistency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    safe_sigma = np.maximum(sigma, 0.0)
    safe_cost = np.maximum(cost, 1e-12)
    
    # UCB for minimization: we want to minimize, so we prefer low mu
    # UCB_min = mu - beta * sigma (lower is better)
    # But acquisition should be higher = better, so:
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Use beta=2.0 for balanced exploration/exploitation
    beta = 2.0
    ucb = beta * safe_sigma - mu
    
    # Divide by linear cost penalty
    score = ucb / safe_cost
    
    # Ensure output is finite and real
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
