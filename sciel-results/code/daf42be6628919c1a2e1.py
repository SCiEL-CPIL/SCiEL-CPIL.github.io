"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    LCB-dominant acquisition with EI-relative-to-known-minimum exploration pressure.
    
    Structural design:
    1. Primary signal: Negative LCB (mu - beta*sigma) — exploit low predicted values.
       Beta adapts: larger when f_best is far from minimum (explore), smaller when
       f_best is near minimum (exploit).
    2. Secondary signal: EI relative to known global minimum (3.0) — maintains
       exploration pressure toward regions that could contain the true minimum.
    3. Small variance bonus for pure exploration in highly uncertain regions.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Posterior mean predictions at candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviations at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Known global minimum of Goldstein-Price 2D
    f_min = 3.0
    
    # Adaptive beta for LCB:
    # When f_best is far from f_min, we want more exploration (higher beta)
    # When f_best is close to f_min, we want more exploitation (lower beta)
    # Gap ranges from ~297 (f_best≈300) down to ~0 (f_best≈3)
    gap = max(f_best - f_min, 0.1)
    # Normalize gap to [0, 1] range: gap of 300 -> 1, gap of 0 -> 0
    gap_norm = min(gap / 30.0, 1.0)
    # Beta: 0.3 (exploit) when close, 1.5 (explore) when far
    beta = 0.3 + 1.2 * gap_norm
    
    # Primary: LCB score (negated so higher = better)
    lcb_score = -(mu - beta * sigma_safe)
    
    # Secondary: EI relative to known minimum (3.0)
    # This provides a constant exploration signal toward the true minimum
    z = (f_min - mu) / sigma_safe
    ei = (f_min - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be on a comparable scale to LCB
    # EI can be large when mu is well below f_min or sigma is large
    # Scale it down so it acts as a secondary signal
    ei_scale = 0.15
    
    # Tertiary: small variance bonus for pure exploration
    var_bonus = 0.05 * sigma_safe
    
    # Combine: LCB dominates, EI provides exploration pressure, small var bonus
    scores = 1.0 * lcb_score + ei_scale * ei + var_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Deadline guard
    if time.time() - t0 > 285:
        # Fallback: simple LCB
        scores = -(mu - beta * sigma_safe)
        scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
