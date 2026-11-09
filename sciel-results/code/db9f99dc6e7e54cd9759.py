import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement per unit cost.
    Score = EI / cost.
    This directly optimizes the rate of improvement relative to the expense.
    """
    if time.time() - _START_TIME > _TIMEOUT:
        # Return a neutral score or just zeros to allow the loop to finish/timeout safely
        return np.zeros(len(mu))

    n = len(mu)
    
    # 1. Calculate Expected Improvement (EI)
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    
    # Standard normal CDF and PDF calculations
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Calculate Cost-Aware Score
    # We want to maximize EI / cost.
    # Cost is in [1, 2] typically, so division is stable.
    # Add small epsilon to cost to be safe against any zero/negative costs (though unlikely).
    cost_safe = np.maximum(cost, 1e-9)
    
    score = ei / cost_safe
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
