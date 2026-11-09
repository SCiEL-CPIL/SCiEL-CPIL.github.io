# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # Leave margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Structural Change:
    Replaces the hybrid EI/UCB mixture with a pure UCB strategy.
    UCB is linear in mu and sigma, providing a different exploration/exploitation
    trade-off compared to the nonlinear EI. This is particularly effective for the
    Beale function's wide flat regions where EI might underestimate the potential
    for large improvements.
    
    We use a constant beta to avoid potential miscalibration of adaptive progress
    metrics. A beta of 1.5 is chosen to balance exploration and exploitation
    for the 2D Beale function.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB for minimization: -mu + beta * sigma
    beta = 1.5
    
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
