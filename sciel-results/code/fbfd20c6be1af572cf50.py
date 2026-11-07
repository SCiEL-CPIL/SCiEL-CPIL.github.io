"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition with adaptive beta.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better for selection)
    """
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: larger when f_best is far from 0 (exploration),
    # smaller when f_best is close to 0 (exploitation)
    # beta = 1 + 10 / max(f_best, eps)
    # But we need to be careful: f_best is the minimum observed value
    # For Beale function, minimum is 0, so f_best > 0 typically
    
    # Use a safer formulation
    f_best_safe = max(abs(f_best), 1e-6)
    
    # Beta decays as we get closer to the optimum
    # When f_best is large (e.g., 100), beta ~ 1 + 0.1 = 1.1 (moderate exploration)
    # When f_best is small (e.g., 0.01), beta ~ 1 + 1000 = 1001 (too much exploration?)
    # Let's think differently: we want MORE exploration when f_best is bad
    
    # Actually, let's use: beta = 1 + c / f_best_safe
    # But cap it to avoid extreme values
    c = 2.0
    beta = 1.0 + c / f_best_safe
    beta = np.clip(beta, 1.0, 50.0)
    
    # LCB: minimize mu - beta * sigma
    # We return scores where higher means "evaluate next"
    # So we return -LCB = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
