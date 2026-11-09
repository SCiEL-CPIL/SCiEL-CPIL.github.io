"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with adaptive exploration.
    
    Uses EI combined with an adaptive exploration bonus that scales with
    the relative uncertainty of each point, plus a log-sigma term for
    smoother exploration pressure.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: scale bonus by how much this point's sigma
    # exceeds the median sigma, encouraging exploration in uncertain regions
    sigma_median = np.median(sigma_safe)
    sigma_relative = sigma_safe / max(sigma_median, 1e-9)
    
    # Exploration bonus: base + relative uncertainty component
    # The 0.08 base is slightly lower than 0.1 to let EI dominate more,
    # but the relative term amplifies exploration where uncertainty is highest
    exploration_bonus = 0.08 * sigma_safe + 0.05 * sigma_relative * sigma_median
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
