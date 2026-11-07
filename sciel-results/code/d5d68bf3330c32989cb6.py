"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition combining UCB exploration with Expected Improvement exploitation.
    
    Early in optimization (high uncertainty relative to improvement gap), favor exploration
    via UCB to map the multimodal landscape. As convergence progresses, shift toward EI 
    for focused exploitation near the global minimum.
    
    This combines:
    - UCB's strong early exploration drive (current program's strength)
    - EI's focused exploitation (inspiration 3's simplicity and effectiveness)
    - Adaptive blending based on optimization progress (inspiration 1&2's machinery)
    """
    # Track elapsed time for safety
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute UCB score for minimization: -mu + k * sigma
    # Higher score = more optimistic about finding a low value
    k_ucb = 2.0  # Slightly higher k for stronger early exploration
    ucb_score = -mu + k_ucb * sigma_safe
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Determine progress indicator
    # Progress = 0: early stage, high uncertainty, large gap -> explore
    # Progress = 1: late stage, low uncertainty, small gap -> exploit
    min_mu = np.min(mu)
    max_sigma = np.max(sigma_safe)
    min_sigma = np.min(sigma_safe)
    
    if max_sigma > min_sigma:
        sigma_range = max_sigma - min_sigma
    else:
        sigma_range = 1e-9
    
    # Gap between current best and best predicted value
    gap = f_best - min_mu
    # Normalize gap by uncertainty scale
    if max_sigma > 1e-9:
        progress = np.clip(gap / (max_sigma + 1e-9), 0.0, 1.0)
    else:
        progress = 1.0
    
    # Blend weights: explore more early, exploit more late
    # Use a smooth sigmoid-like transition
    exploration_weight = 0.6 * (1.0 - progress)
    exploitation_weight = 1.0 - exploration_weight
    
    # Normalize scores to [0, 1] for blending
    ucb_max = np.max(ucb_score)
    ucb_min = np.min(ucb_score)
    if ucb_max > ucb_min:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(ucb_score)
    
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei_score)
    
    # Final blended score
    score = exploitation_weight * ei_norm + exploration_weight * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
