"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 10.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time) - _MARGIN

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using multiplicative Probability of Improvement
    with saturating sigma boost, divided by cost^alpha.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    The multiplicative form PI * (1 + beta * sigma / (sigma + k)) couples exploitation
    and exploration with a saturating exploration term. Dividing by cost^alpha makes
    it cost-aware with tunable sensitivity.
    """
    # Deadline guard
    if _remaining_time() < 1.0:
        # Fast fallback
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        beta = 0.5
        k = 0.1
        sat = sigma_safe / (sigma_safe + k)
        scores = pi * (1.0 + beta * sat)
        cost_safe = np.maximum(cost, 1e-8)
        alpha = 1.5
        scores = scores / (cost_safe ** alpha)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Probability of Improvement (PI)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Saturating exploration boost: sigma / (sigma + k) approaches 1 as sigma -> inf
    # This prevents over-exploration in regions with very high uncertainty
    beta = 0.5
    k = 0.1
    sat = sigma_safe / (sigma_safe + k)
    scores = pi * (1.0 + beta * sat)
    
    # Cost safety
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware: divide by cost^alpha with alpha > 1 for stronger cost preference
    alpha = 1.5
    scores = scores / (cost_safe ** alpha)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
