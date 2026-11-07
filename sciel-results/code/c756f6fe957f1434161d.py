import numpy as np
import time

_start_time = None
_deadline = 300.0
_safety_margin = 5.0

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check if we are running out of time
    if time.time() > _start_time + _deadline - _safety_margin:
        # Fallback: Deterministic score based on mean
        # Minimizing objective, so lower mu is better.
        # Higher cost is worse.
        cost_safe = np.maximum(cost, 1e-8)
        fallback_scores = -mu / cost_safe
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
        
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling Strategy
    # Sample a value from the GP posterior for each candidate
    # y ~ N(mu, sigma^2)
    z = np.random.randn(len(mu))
    y_samples = mu + sigma_safe * z
    
    # We want to minimize the objective value y.
    # The evaluation cost is 'cost'.
    # We define a score to be maximized.
    # A good candidate has low y and low cost.
    # Score = -y / cost
    # This implicitly balances exploration (via y_samples variance) and exploitation (via mu)
    # and adjusts for cost.
    
    scores = -y_samples / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
