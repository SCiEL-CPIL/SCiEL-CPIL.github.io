"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Enforce deadline with safety margin
    start_time = time.time()
    deadline = start_time + 295.0  # 300s limit, 5s safety margin

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zeros in sigma to avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-scores
    z = (f_best - mu) / sigma_safe
    
    # Pre-compute CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (uncosted)
    EI = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Power-law cost penalty: softer than linear, allows exploration in slightly higher-cost regions
    # cost^0.7 is less punitive than cost^1.0 for high-cost regions
    cost_penalty = np.power(cost, 0.7)
    
    # Base cost-aware EI
    score_base = EI / cost_penalty
    
    # Exploration bonus: add uncertainty-driven exploration when EI is low
    # This helps escape local optima and find better minima across instances
    # Normalize EI to [0, 1] range for blending
    ei_min = np.min(EI)
    ei_max = np.max(EI)
    ei_range = ei_max - ei_min
    if ei_range < 1e-10:
        ei_normalized = np.ones_like(EI) * 0.5
    else:
        ei_normalized = (EI - ei_min) / ei_range
    
    # Exploration weight: higher when EI is low (more exploration needed)
    exploration_weight = 1.0 - ei_normalized
    
    # Uncertainty-based exploration score
    # Use sigma normalized by the scale of the problem
    sigma_scale = np.max(sigma_safe) + 1e-10
    exploration_score = (sigma_safe / sigma_scale) * exploration_weight
    
    # Blend: base score + exploration bonus
    # The exploration bonus is scaled to be comparable to the base score
    base_scale = np.max(score_base) + 1e-10
    exploration_bonus = 0.3 * exploration_score * base_scale
    
    score = score_base + exploration_bonus
    
    # Safety: ensure all scores are finite
    if not np.all(np.isfinite(score)):
        # Fallback to simple cost-aware EI
        score_fallback = EI / np.maximum(cost, eps)
        if np.all(np.isfinite(score_fallback)):
            score = score_fallback
        else:
            score = np.zeros_like(mu)
    
    # Check time budget
    if time.time() > deadline:
        # Return whatever we have
        pass
    
    return score
# EVOLVE-BLOCK-END
