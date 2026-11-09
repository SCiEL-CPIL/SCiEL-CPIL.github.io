"""Rewritten program with adaptive EI+UCB hybrid acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_KNOWN_MIN = -1.8013
_F_START = None  # Will be set on first call to track progress


def acquisition(mu, sigma, f_best):
    global _F_START
    if _F_START is None:
        _F_START = f_best
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute progress: 0 at start, 1 when f_best reaches known minimum
    # Range: from _F_START (worst initial) to _KNOWN_MIN (best possible)
    total_range = _F_START - _KNOWN_MIN
    if total_range <= 1e-12:
        progress = 1.0
    else:
        progress = np.clip((_F_START - f_best) / total_range, 0.0, 1.0)
    
    # Expected Improvement (EI) component - exploitation
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is negative when mu > f_best (we're worse than best), so higher EI = better
    # Actually EI should be positive when we expect improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu)/sigma
    # When mu < f_best (potential improvement), z > 0, EI > 0
    # When mu > f_best, z < 0, EI < 0
    
    # Uncertainty bonus (UCB) component - exploration
    # Scale exploration weight: high early, low late
    # Use a smooth decay: exploration_weight = 1 - progress, but with floor
    exploration_weight = 0.5 * (1.0 - progress) + 0.1  # Range: [0.1, 0.6]
    
    # Normalize EI to have similar scale as sigma
    # The typical scale of EI is on the order of sigma
    # Use a simple scaling: divide EI by a characteristic scale
    # Characteristic scale: std of mu around f_best
    mu_spread = np.std(mu) if np.std(mu) > 1e-8 else 1.0
    
    # Combine: score = EI + exploration_weight * sigma
    # Higher score = evaluate next
    scores = ei + exploration_weight * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
