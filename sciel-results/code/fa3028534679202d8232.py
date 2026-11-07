"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    weighted exploration term based on posterior uncertainty (sigma).
    
    The exploration term helps escape local optima by encouraging
    evaluation in regions of high uncertainty, which is beneficial for
    multimodal functions like Styblinski-Tang 1D.
    """
    # Clip sigma to avoid division by zero or numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (standard)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage high-uncertainty regions
    # The weight is small enough not to dominate EI but large enough to
    # provide meaningful exploration
    exploration_weight = 0.15
    exploration = exploration_weight * sigma_safe
    
    # Combined score: EI + exploration bonus
    score = ei + exploration
    
    # Ensure finite values (defensive)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
