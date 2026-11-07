import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 325:
        # Fallback to EI if time is tight
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei

    # Adaptive UCB: beta decays from 4.0 to 1.5 over 25 acquisition calls
    # This provides stronger early exploration for multimodal Branin landscape
    # while converging to more exploitation-focused behavior
    if _call_count <= 25:
        progress = (_call_count - 1) / 24.0  # 0 to 1
        beta = 4.0 - 2.5 * progress  # 4.0 to 1.5
    else:
        beta = 1.5
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: mu - beta * sigma
    ucb = mu - beta * sigma_safe
    
    # Also compute EI as a fallback/blend
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to comparable scale with UCB
    # Use the range of EI to normalize
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    # Normalize UCB to [0, 1] range
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-12
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: weight more toward UCB early, more toward EI later
    # In early phase (high beta), UCB dominates; in late phase, EI dominates
    blend_weight = min(1.0, _call_count / 15.0)  # 0 to 1 over first 15 calls
    score = (1 - blend_weight) * ucb_norm + blend_weight * ei_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
