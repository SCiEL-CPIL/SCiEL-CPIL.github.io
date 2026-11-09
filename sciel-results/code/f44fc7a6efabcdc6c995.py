"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 300s limit, keep 10s margin


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a safe fallback
    if time.time() > _DEADLINE:
        # Fallback: return cost-aware EI with minimal computation
        z = (f_best - mu) / (sigma + 1e-6)
        scores = ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / (cost + 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Expected Improvement per unit cost
    # Use numerical safeguards
    sigma_safe = np.maximum(sigma, 1e-6)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid extreme values in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    scores = ei / (cost + 1e-9)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
