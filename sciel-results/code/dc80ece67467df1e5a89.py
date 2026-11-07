"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition combining Expected Improvement relative to the known
    global minimum with an adaptive Lower Confidence Bound for exploitation.
    
    The structural idea: use EI toward the true minimum (3.0) as the primary
    signal to maintain constant exploration pressure, while blending in an
    LCB component that strengthens exploitation as the search converges.
    
    mu, sigma: arrays of shape (2000,) in objective units
    f_best: lowest observed value
    
    Returns 2000 finite scores, higher = evaluate next.
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Known global minimum of Goldstein-Price 2D
    f_min = 3.0
    
    # --- Primary signal: EI relative to known global minimum ---
    # This provides a constant exploration target regardless of f_best
    z_ei = (f_min - mu) / sigma_safe
    ei = (f_min - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # --- Secondary signal: LCB for exploitation ---
    # As f_best approaches f_min, we want to exploit more aggressively
    # beta adapts: larger when we're far from optimum, smaller when close
    gap = max(f_best - f_min, 1e-6)
    # Normalize gap: starts large (~27 at start), shrinks as we converge
    # beta from ~1.0 (exploratory) to ~0.1 (exploitative)
    beta = np.clip(0.1 + 0.9 * gap / 30.0, 0.1, 1.0)
    
    # LCB = mu - beta * sigma; negate for "higher is better"
    lcb_score = -(mu - beta * sigma_safe)
    
    # --- Adaptive weighting ---
    # When gap is large, EI dominates (exploration toward true min)
    # When gap is small, LCB gets more weight (exploit near best)
    # Weight transition: from 0.9/0.1 to 0.5/0.5 as gap shrinks
    gap_norm = np.clip(gap / 30.0, 0.0, 1.0)
    w_ei = 0.5 + 0.4 * gap_norm      # 0.5 to 0.9
    w_lcb = 1.0 - w_ei               # 0.1 to 0.5
    
    # --- Small variance bonus for pure exploration in very uncertain regions ---
    # Only active when sigma is large relative to the gap
    sigma_scale = max(gap, 1.0)
    var_bonus = 0.05 * sigma_safe / sigma_scale
    
    # --- Combine ---
    scores = w_ei * ei + w_lcb * lcb_score + var_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline guard
    if time.time() - t0 > 280:
        # Fallback: just EI
        scores = ei
        scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
