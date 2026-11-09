"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 10s safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition blending EI, UCB, and PI, normalized by cost.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) array of acquisition scores (higher = more desirable)
    """
    # Safety: ensure finite outputs
    if _check_deadline():
        # Return a simple, safe fallback
        return np.zeros_like(mu, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Upper Confidence Bound (UCB) - exploration
    # Use a beta that balances exploration; 2.0 is a reasonable default
    beta = 2.0
    ucb = mu + beta * sigma
    
    # Probability of Improvement (PI)
    pi = norm.cdf((f_best - mu) / sigma)
    
    # Blend: 0.5 * EI + 0.3 * UCB + 0.2 * PI
    # Normalize UCB and PI to be comparable to EI
    # EI is in objective units, UCB is in objective units, PI is in [0,1]
    # Scale PI by the range of f_best - mu to make it comparable
    scale = np.maximum(np.abs(f_best - mu), 1e-9)
    pi_scaled = pi * scale
    
    blended = 0.5 * ei + 0.3 * ucb + 0.2 * pi_scaled
    
    # Normalize by cost (higher cost = lower score)
    cost_safe = np.maximum(cost, 1e-9)
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
