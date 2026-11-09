import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses Probability of Improvement (PI) normalized by the square root of the cost.
    This provides a smoother gradient for exploration in multimodal landscapes
    compared to Expected Improvement, and balances exploration efficiency
    against computational budget.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Probability of Improvement (for minimization)
    improvement = f_best - mu
    z = improvement / sigma
    pi = norm.cdf(z)
    
    # Cost Adjustment: divide by cost^0.5 for smoother cost normalization
    score = pi / np.sqrt(cost)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
