"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = 325.0  # safety margin before 330s kill
_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Gaussian EI with an exploration bonus.
    
    The exploration bonus (0.1 * sigma) prevents over-convergence in early
    stages with only 5 initial samples, promoting coverage of the 6D space.
    This addresses the under-exploration observed with pure EI or t-EI
    in this limited-budget regime.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return scores that favor points with high uncertainty (exploration)
        # to at least return finite values
        return sigma
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard Gaussian Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus proportional to uncertainty
    # This encourages sampling in regions of high sigma, which is critical
    # when only 5 initial points have been observed in 6D space
    exploration_bonus = 0.1 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
