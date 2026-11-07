"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 320.0
_CALL_COUNT = 0

def _norm_cdf(x):
    return norm.cdf(x)

def acquisition(mu, sigma, f_best, cost):
    global _CALL_COUNT, _DEADLINE, _START_TIME
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        return np.ones_like(cost) / cost

    _CALL_COUNT += 1
    
    # Clip sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Estimate progress: 
    # Start with 5 points. Budget 30. Avg cost ~1.5.
    # 5 * 1.5 = 7.5. Remaining 22.5.
    # 22.5 / 1.5 = 15 evaluations.
    # So total evaluations ~ 20.
    # Progress p goes from 0 to 1 over ~15 calls in the loop.
    # We are in the loop, so _CALL_COUNT starts at 1 for the first loop iteration.
    p = min(1.0, _CALL_COUNT / 15.0)
    
    # Dynamic Beta: High exploration early, low later
    beta_max = 3.0
    beta_min = 0.5
    beta = beta_max * (1.0 - p) + beta_min * p
    
    # UCB for minimization: -mu - beta * sigma
    ucb = -mu - beta * sigma_safe
    
    # PI: Probability that f(x) < f_best
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    pi = _norm_cdf(z)
    
    # Combine UCB and PI
    # UCB provides the "push" to low mean/high uncertainty areas.
    # PI provides a smooth gradient towards improvement.
    # Weighting: UCB is more aggressive.
    w_ucb = 0.8
    w_pi = 0.2
    raw_score = w_ucb * ucb + w_pi * pi
    
    # Cost-aware: Divide by cost
    cost_safe = np.maximum(cost, 1e-8)
    scores = raw_score / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Tiny noise for tie-breaking
    idx = np.arange(len(scores))
    scores += 1e-10 * np.sin(idx * 0.1)
    
    return scores
# EVOLVE-BLOCK-END
