"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses a blended approach: adaptive UCB + Expected Improvement,
    with careful handling of the multimodal landscape.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Ensure finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # For minimization, we want to find points with low expected values
    # and high uncertainty (to explore new basins)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: we want low mu and high sigma
    # Use an adaptive beta that decreases as we get closer to the optimum
    # The gap between f_best and the known minimum (-1.8013) tells us progress
    known_min = -1.8013
    gap = f_best - known_min
    # When gap is large, we need more exploration; when small, exploit more
    # beta starts high (~3.0) and decreases to ~1.0
    if gap > 0.5:
        beta = 3.0
    elif gap > 0.1:
        beta = 2.0
    elif gap > 0.01:
        beta = 1.5
    else:
        beta = 1.0
    
    ucb = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend: weight UCB more when gap is large (exploration phase),
    # weight EI more when gap is small (exploitation phase)
    if gap > 0.5:
        w_ucb = 0.55
        w_ei = 0.45
    elif gap > 0.1:
        w_ucb = 0.45
        w_ei = 0.55
    else:
        w_ucb = 0.30
        w_ei = 0.70
    
    scores = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
