"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) with a
    lower-confidence-bound (LCB) style exploration term, tuned for the
    Beale 2D function on [-4.5, 4.5]^2.
    
    mu: shape (2000,) posterior mean in objective units
    sigma: shape (2000,) posterior std in objective units
    f_best: best observed value so far (lowest)
    
    Returns: shape (2000,) scores, higher = more promising to evaluate next.
    """
    # Ensure finite inputs; guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # LCB: we want low mu and high sigma (exploration)
    # The LCB value is mu - xi * sigma; we want to maximize (f_best - LCB)
    # which is (f_best - mu + xi * sigma)
    # But EI already captures exploitation. Let's add a mild exploration bonus.
    # A common blend: score = EI + beta * sigma / (sigma + eps)
    # Or use a weighted sum of normalized EI and exploration term.
    
    # Let's use a simpler robust blend:
    # score = EI + 0.1 * sigma (encourages exploring high-uncertainty regions)
    # But we need to be careful with scale. Since mu and sigma are in objective units,
    # and Beale's range is roughly [0, ~60], sigma might be small.
    
    # Alternative: use a probability-weighted improvement with an exploration floor.
    # Let's try: score = EI + 0.05 * sigma_safe
    
    # To make it more adaptive, we can scale the exploration term by the current best.
    # But f_best can be 0, so let's use a fixed small coefficient.
    
    score = ei + 0.05 * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
