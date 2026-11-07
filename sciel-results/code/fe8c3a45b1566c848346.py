"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_global_start_time = None
_global_eval_count = 0


def acquisition(mu, sigma, f_best):
    global _global_start_time, _global_eval_count
    
    if _global_start_time is None:
        _global_start_time = time.time()
    
    _global_eval_count += 1
    
    # Compute elapsed time
    elapsed = time.time() - _global_start_time
    
    # Deadline guard: 330s total, keep safety margin
    # If we're running out of time, just return a simple score
    if elapsed > 310:
        # Fallback: simple EI
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Expected Improvement
    z_ei = (f_best - mu) / (sigma + 1e-9)
    ei_scores = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # UCB with adaptive beta
    # Early on, we want more exploration; later, more exploitation
    # f_best starts high (branin can be ~30) and decreases toward 0.398
    # Use a beta that decreases as we get closer to the optimum
    if f_best > 10:
        beta = 2.0
    elif f_best > 5:
        beta = 1.5
    elif f_best > 2:
        beta = 1.0
    elif f_best > 1:
        beta = 0.5
    else:
        beta = 0.3
    
    ucb_scores = -mu + beta * sigma
    
    # Normalize both to [0, 1] for blending
    ei_min, ei_max = np.min(ei_scores), np.max(ei_scores)
    ucb_min, ucb_max = np.min(ucb_scores), np.max(ucb_scores)
    
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei_scores - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei_scores)
    
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb_scores - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_scores)
    
    # Blend: early on, more UCB; later, more EI
    if _global_eval_count <= 10:
        weight_ei = 0.3
    elif _global_eval_count <= 20:
        weight_ei = 0.5
    else:
        weight_ei = 0.7
    
    blended = weight_ei * ei_norm + (1 - weight_ei) * ucb_norm
    
    # Ensure finite
    blended = np.where(np.isfinite(blended), blended, 0.0)
    
    return blended
# EVOLVE-BLOCK-END
