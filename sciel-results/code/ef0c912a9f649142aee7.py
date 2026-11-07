# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Probability of Improvement (PI).
    Score = PI(x) / cost(x).
    Higher score indicates a more promising candidate to evaluate next.
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # Safety margin

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: Return uniform random or simple heuristic
        # To ensure validity, return ones (all equal) or zeros.
        # Returning ones allows the loop to proceed if it picks the first one.
        return np.ones(len(mu))

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate Z-scores
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    # P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Cost-Aware Score
    # We want to maximize the probability of finding a better point per unit cost.
    scores = pi / cost_safe
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
