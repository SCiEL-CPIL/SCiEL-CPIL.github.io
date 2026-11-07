"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI) per unit cost.
    
    This function computes the probability that a candidate point improves upon the
    best observed value, normalized by the evaluation cost. This approach is often
    more robust than Expected Improvement (EI) when costs vary, as it directly
    measures the likelihood of success relative to the expense.
    
    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate
                candidates that are more likely to be evaluated next.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit with 20s safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    N = len(mu)
    
    # Safety check for sigma
    # If sigma is 0, the probability is 1 if mu < f_best, else 0.
    # norm.cdf handles this, but division by zero in z-score must be avoided.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Probability of Improvement (POI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Cost-aware score: POI divided by cost
    # We add a small epsilon to cost to avoid division by zero, though costs are >= 1
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = poi / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (unlikely to be hit for 2000 points, but good practice)
    if time.time() > deadline:
         return scores
        
    return scores
# EVOLVE-BLOCK-END
