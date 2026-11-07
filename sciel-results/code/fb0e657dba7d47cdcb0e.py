import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining LCB with cost normalization.
    
    LCB (Lower Confidence Bound) for minimization: mu - beta * sigma
    We want to minimize this, so we maximize its negative.
    Dividing by cost penalizes expensive evaluations.
    
    The LCB approach with beta=2 provides stronger exploration than EI's
    implicit balance, helping escape local minima in multi-modal Styblinski-Tang.
    The cost division naturally biases toward center-of-domain points near
    shifted minima.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu_safe / cost_safe
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # So we maximize -(mu - beta * sigma) = -mu + beta * sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # Cost adjustment: divide by cost to penalize expensive points
    # This biases search toward the center where cost is lower
    final_score = lcb_score / cost
    
    # Ensure finite output
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
