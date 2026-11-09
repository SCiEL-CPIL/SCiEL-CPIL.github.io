"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """Acquisition function combining Expected Improvement and a diversity bonus."""
    # Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a diversity bonus based on uncertainty to encourage exploration
    # This helps escape local minima in the Michalewicz function
    diversity_bonus = 0.1 * sigma_safe
    
    # Combine EI and diversity
    score = ei + diversity_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
