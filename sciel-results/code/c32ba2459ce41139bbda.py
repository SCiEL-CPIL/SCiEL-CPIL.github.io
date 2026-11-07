"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-inspired acquisition for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Adaptive uncertainty bonus for exploration
    
    The Beale function is multi-modal, so we need to balance
    exploitation of known good regions with exploration of uncertain areas.
    """
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: use sigma directly but scaled by a factor
    # that depends on how much improvement is possible
    # When f_best is close to 0 (minimum), we want less exploration
    # When f_best is far from 0, we want more exploration
    
    # Scale exploration based on current best
    # Beale minimum is 0, typical values are in [0, ~15]
    improvement_scale = np.clip(f_best, 0.1, 10.0)
    
    # Exploration bonus: proportional to uncertainty, scaled by improvement potential
    # Use a softer scaling to avoid too aggressive exploration
    exploration_bonus = 0.15 * sigma_safe * np.sqrt(improvement_scale)
    
    # Combined score
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
