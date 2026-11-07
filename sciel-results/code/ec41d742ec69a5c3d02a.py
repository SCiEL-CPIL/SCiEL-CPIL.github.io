import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with normalized exploration and soft cost penalty.
    
    Args:
        mu: Array of GP means (2000,)
        sigma: Array of GP standard deviations (2000,)
        f_best: Scalar, lowest observed function value
        cost: Array of evaluation costs (2000,)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs to ensure numerical stability
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in cdf for extreme values
    z = np.clip(z, -10, 10)
    pi = norm.cdf(z)
    
    # Calculate normalized exploration term
    # This makes the exploration bonus scale-invariant relative to the current GP uncertainty
    sigma_max = np.max(sigma)
    if sigma_max > 1e-8:
        exploration = sigma / sigma_max
    else:
        exploration = np.zeros_like(sigma)
        
    # Combine PI and exploration
    # Beta balances exploitation (PI) and exploration (uncertainty)
    # PI is in [0, 1], exploration is in [0, 1]
    # Beta=1.0 gives equal weight to exploration and exploitation
    beta = 1.0
    base_score = pi + beta * exploration
    
    # Apply soft cost penalty
    # Cost is in [1, 2]. 
    # cost^-0.5 maps [1, 2] to [1.0, 0.707]
    # This multiplicatively discounts expensive points without zeroing them out
    cost_penalty = np.power(cost, -0.5)
    
    scores = base_score * cost_penalty
    
    # Deadline guard: return safe fallback if computation is unexpectedly slow
    # The computation is O(N) and very fast, but we adhere to the directive.
    if time.time() - t_start > 2.0:
        return exploration
        
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
