"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration, plus a safety margin for numerical stability.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component with adaptive exploration
    # Use a moderate exploration constant; since we have 2000 candidates per step
    # and ~25 steps total, we want balanced exploration
    beta = 0.5
    ucb = -mu + beta * sigma  # Note: we want to maximize, so higher score = better
    # Actually, for minimization: UCB score = -mu + beta * sigma
    # But EI already handles the tradeoff. Let's blend.
    
    # Normalize EI to [0, 1] range approximately for blending
    # Find max EI for normalization
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # UCB normalized
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend: primarily EI with some UCB for exploration
    # Weight shifts from more exploration early to more exploitation later
    # But we don't have step info, so use a fixed blend
    alpha = 0.8  # 80% EI, 20% UCB
    score = alpha * ei_norm + (1 - alpha) * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
