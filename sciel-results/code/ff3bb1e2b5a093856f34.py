"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (POI) for minimization.
    
    Strategy:
    - POI scores candidates by the probability that they will be better than f_best.
    - For minimization, P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    - This provides a bounded [0,1] score that emphasizes candidates likely to improve.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # POI for minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
