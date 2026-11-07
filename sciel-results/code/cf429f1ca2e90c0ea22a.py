import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    divided by cost^gamma.
    
    Score = (beta * sigma - mu) / cost^gamma
    Higher score means more likely to be evaluated next.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # UCB parameter
    # Beta controls the trade-off between exploitation and exploration.
    # A higher beta encourages more exploration.
    # For 6D Hartmann, a moderate to high beta is often beneficial.
    beta = 2.5
    
    # Cost penalty exponent
    # Gamma controls how strongly we penalize higher cost evaluations.
    # The optimum is in the low-cost region, so a significant penalty is good.
    gamma = 1.5
    
    # Calculate UCB score for minimization
    # We want to minimize f(x).
    # UCB for minimization: minimize mu - beta * sigma
    # To make it a "higher is better" score: score = beta * sigma - mu
    ucb_score = beta * sigma - mu
    
    # Apply cost penalty
    cost_penalty = np.power(cost, gamma)
    
    score = ucb_score / cost_penalty
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
         return np.zeros_like(mu)

    return score
