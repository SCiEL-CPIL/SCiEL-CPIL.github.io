"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Hartmann 3D.
    Uses a hybrid approach: standard Expected Improvement (EI) combined with
    a small, adaptive exploration bonus to help escape local optima early on.
    """
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add a small exploration bonus.
    # The bonus is scaled by the relative uncertainty compared to the current
    # improvement potential. This encourages exploration when the model is
    # uncertain but tapers off as we get closer to the optimum or as
    # uncertainty decreases.
    # A small weight (0.05) ensures we don't deviate too far from pure EI.
    exploration_weight = 0.05
    # Normalize exploration bonus by the range of EI to keep it stable
    ei_range = np.max(ei) - np.min(ei)
    if ei_range > 1e-12:
        exploration_bonus = exploration_weight * (sigma_safe / ei_range) * np.max(ei)
    else:
        exploration_bonus = 0.0
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
