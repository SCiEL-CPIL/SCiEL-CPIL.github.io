"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with adaptive exploration.
    
    Structural change: Instead of a fixed exploration weight, we use an adaptive
    weight that scales with the current best value. When f_best is large (early
    optimization), we explore more aggressively. When f_best is small (late optimization),
    we focus on exploitation.
    
    This is a structural change from the parent's fixed PI-weighted approach to an
    adaptive scaling that responds to the optimization progress.
    """
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration weight:
    # - When f_best is large (early), we want more exploration
    # - When f_best is small (late), we want less exploration
    # Beale function minimum is 0, typical initial values can be ~10-15
    # Scale exploration weight inversely with f_best, but with a floor to prevent
    # complete shutdown of exploration
    improvement_potential = np.clip(f_best, 0.5, 10.0)
    
    # Adaptive weight: higher when improvement potential is high
    # Use a smooth decay: weight ~ 1 / (1 + f_best)
    adaptive_weight = 0.3 / (1.0 + 0.5 * f_best)
    
    # PI-weighted exploration: only explore where improvement is likely
    pi_weighted_exploration = adaptive_weight * pi * sigma_safe
    
    # Small raw sigma term to ensure baseline diversity
    raw_exploration = 0.05 * sigma_safe
    
    scores = ei + pi_weighted_exploration + raw_exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
