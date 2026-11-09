import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_call_count = 0
_prev_f_best = None

def acquisition(mu, sigma, f_best):
    global _call_count, _prev_f_best

    # Time guard: If we are close to the limit, return a simple score to avoid timeout
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu

    # Detect new run (new seed) if f_best increases significantly
    if _prev_f_best is not None and f_best > _prev_f_best + 1e-6:
        _call_count = 0
    _prev_f_best = f_best
    _call_count += 1

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (f_best - mu) / sigma
        # Avoid division by zero issues
        z = np.where(sigma > 1e-8, z, 0.0)
        
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        ei = (f_best - mu) * Phi_z + sigma * phi_z
        # EI should be non-negative
        ei = np.maximum(ei, 0.0)
        ei = np.where(np.isfinite(ei), ei, 0.0)

    # Normalize EI and Sigma to [0, 1]
    max_ei = np.max(ei)
    max_sigma = np.max(sigma)
    
    if max_ei > 1e-8:
        ei_norm = ei / max_ei
    else:
        ei_norm = np.zeros_like(ei)
        
    if max_sigma > 1e-8:
        sigma_norm = sigma / max_sigma
    else:
        sigma_norm = np.zeros_like(sigma)

    # Dynamic Weight: Increase weight on EI (exploitation) as evaluations progress
    # w_ei starts low (exploration) and increases to high (exploitation)
    w_ei = 1.0 - np.exp(-0.15 * _call_count)
    # Clip to ensure some exploration always remains
    w_ei = np.clip(w_ei, 0.1, 0.9)
    w_unc = 1.0 - w_ei

    # Weighted sum of normalized objectives
    scores = w_ei * ei_norm + w_unc * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
