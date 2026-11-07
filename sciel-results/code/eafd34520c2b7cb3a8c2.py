"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a blend of Expected Improvement and an uncertainty-based exploration term,
    with adaptive weighting based on progress toward the known global minimum.
    """
    t0 = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Progress toward the known global minimum (-1.8013)
    global_min = -1.8013
    # progress: 0 when f_best is poor, 1 when f_best is near global_min
    # Use a reasonable range for f_best
    f_best_start = 0.5  # typical starting value
    progress = np.clip((f_best - f_best_start) / (global_min - f_best_start), 0.0, 1.0)
    
    # Adaptive exploration weight: high early, low late
    # Start with strong exploration, transition to exploitation
    beta = 2.0 - 1.5 * progress  # beta from 2.0 to 0.5
    
    # UCB-like exploration term
    exploration = beta * sigma_safe
    
    # Normalize both terms to comparable scale
    ei_max = np.max(ei) if np.any(ei > 0) else 1.0
    ei_min = np.min(ei) if ei.size > 0 else 0.0
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_norm = (ei - ei_min) / (ei_range + 1e-12)
    
    exp_max = np.max(exploration) if exploration.size > 0 else 1.0
    exp_min = np.min(exploration) if exploration.size > 0 else 0.0
    exp_range = exp_max - exp_min if exp_max > exp_min else 1.0
    exp_norm = (exploration - exp_min) / (exp_range + 1e-12)
    
    # Weighted blend: shift from exploration to exploitation
    w_exploit = 0.3 + 0.7 * progress  # 0.3 early, 1.0 late
    w_explore = 1.0 - w_exploit * 0.6  # 0.82 early, 0.4 late
    
    score = w_exploit * ei_norm + w_explore * exp_norm
    
    # Deadline guard
    if time.time() - t0 > DEADLINE:
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return score
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
