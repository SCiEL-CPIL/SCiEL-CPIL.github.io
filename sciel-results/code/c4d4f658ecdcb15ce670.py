"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with numerical safety and deadline guard.
    t0 = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute expected improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy.stats for numerical stability
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we've spent too much time, return immediately
    # The task says each evaluation must finish within 300s, but we have a 330s wall-clock limit
    # This is a simple acquisition function, so it should be fast.
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
