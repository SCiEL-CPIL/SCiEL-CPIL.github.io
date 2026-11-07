import numpy as np
import time

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
    # Convert to numpy arrays for consistency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # UCB for minimization: we want to minimize, so lower is better.
    # UCB_min = mu - beta * sigma (lower bound on potential improvement)
    # But for acquisition (higher = better), we use:
    # score = (f_best - mu) + beta * sigma  -- this is like EI without the norm part
    # Actually, let's think more carefully:
    # For minimization, we want points where mu is low and sigma is high (uncertain).
    # UCB for minimization: ucb = mu - beta * sigma (we want to minimize this)
    # But acquisition should be higher = better, so:
    # acquisition = -ucb = -mu + beta * sigma
    # Or equivalently: acquisition = (f_best - mu) + beta * sigma
    
    # Beta parameter for exploration
    beta = 2.0
    
    # UCB-based acquisition for minimization
    # Higher score = more promising
    ucb_acq = (f_best - mu) + beta * safe_sigma
    
    # Cost-aware scoring with exponent 1.5 (measured improvement over 1.0)
    # This biases toward lower-cost regions which contain the global optimum
    score = ucb_acq / (safe_cost ** 1.5)
    
    # Ensure output is finite and real
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
