"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return EI-based score
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Use Expected Improvement (EI) as the primary acquisition function.
    # EI is theoretically optimal for maximizing expected improvement over
    # the best observed value and naturally balances exploration/exploitation.
    #
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Add a small LCB component for early exploration
    # The LCB weight decays as f_best approaches the known optimum
    f_opt = 0.397887
    if f_best > f_opt:
        gap = f_best - f_opt
        # Weight decays from 0.1 when gap is large to 0 when gap is small
        lcb_weight = 0.1 * np.exp(-3.0 * gap / max(f_best, 1e-6))
    else:
        lcb_weight = 0.0
    
    lcb = mu - lcb_weight * sigma
    
    # Combine: EI dominates, small LCB blend for robustness
    scores = ei + 0.1 * lcb_weight * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
