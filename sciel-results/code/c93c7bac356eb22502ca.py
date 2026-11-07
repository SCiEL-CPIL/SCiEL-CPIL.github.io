"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 295.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Returns 2000 scores (higher = evaluate next) for 2000 candidate points.
    Uses EI / cost as the primary signal, with numerical safeguards.
    """
    # Deadline guard: return a simple fallback if we're running out of time
    if time.time() > _DEADLINE:
        # Fallback: prefer low mu with cost awareness
        return np.maximum(f_best - mu, 0.0) / (cost + 1e-9)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for expected improvement
    # z = (f_best - mu) / sigma; when z is large positive, EI ≈ f_best - mu
    # when z is large negative, EI ≈ 0
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -15, 15)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: improvement per unit cost
    # cost is in [1, 2] since cost = 1 + mean(x_scaled) where x_scaled in [0,1]
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
