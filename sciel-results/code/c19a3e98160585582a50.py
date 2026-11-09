# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds before the hard limit (300s)

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 295.0  # 300s limit, 5s safety margin
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time())

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) acquisition function.
    
    Strategy:
    1. Calculate Expected Improvement (EI) for minimization:
       EI = (f_best - mu) * Phi(z) + sigma * phi(z)
       where z = (f_best - mu) / sigma, Phi is CDF, phi is PDF of standard normal.
    2. Adjust for cost: The task defines cost as 1 + mean(x_scaled).
       Higher cost implies the point is "further" from the origin (in scaled space) 
       or simply more expensive. We divide the expected gain by the cost to 
       maximize "improvement per unit cost".
    3. This structural change from LCB (linear combination) to EI (non-linear, 
       probability-weighted) better handles the trade-off between exploitation 
       (high probability of improvement) and exploration (high magnitude of 
       potential improvement), while explicitly penalizing expensive evaluations.
    """
    # Check time remaining to avoid timeout
    if _time_remaining() < 0.5:
        # Return a safe, finite score if time is almost up
        return np.zeros(len(mu))
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # z-score for the improvement
    # If sigma is very small, z can be large. Clip to avoid overflow in exp/erf
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    
    # Phi(z) = CDF of standard normal
    # phi(z) = PDF of standard normal
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement
    # If z is very negative (mu >> f_best), EI is 0.
    # If z is very positive (mu << f_best), EI is approx f_best - mu.
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Cost adjustment
    # cost is provided as input. We assume it's positive.
    # We want to maximize EI / cost.
    # To prevent division by zero or near-zero, add a small epsilon.
    cost_safe = np.maximum(cost, 1e-6)
    
    score = EI / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
