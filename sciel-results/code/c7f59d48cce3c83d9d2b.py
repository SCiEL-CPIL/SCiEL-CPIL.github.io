"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive LCB acquisition function for minimization.
    
    score = -mu + beta * sigma
    
    beta adapts based on progress: when f_best is large (early), explore more;
    when f_best is small (late), exploit more.
    """
    # Ensure sigma is non-negative and handle near-zero values
    sigma = np.maximum(sigma, 1e-10)
    
    # Infer progress: f_best ranges from ~25 (worst) to 0 (best)
    # Normalize progress: 0 = start (f_best ~ 25), 1 = end (f_best ~ 0)
    # Use a smooth mapping
    # When f_best is large, we want high beta (explore)
    # When f_best is small, we want low beta (exploit)
    
    # Scale: f_best / 25 gives roughly 1 at start, 0 at end
    # But f_best could be slightly negative due to noise, clamp it
    f_best_clamped = max(f_best, 0.0)
    progress = 1.0 - f_best_clamped / 25.0
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta: start high (~4) for exploration, decrease to low (~0.5) for exploitation
    # Use a smooth decay
    beta_start = 4.0
    beta_end = 0.5
    beta = beta_start - (beta_start - beta_end) * progress
    
    # LCB score for minimization: we want to maximize this
    # Lower mu is better, higher sigma is better (uncertainty bonus)
    score = -mu + beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
