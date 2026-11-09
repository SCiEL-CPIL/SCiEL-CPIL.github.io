"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Pareto-frontier guided exploration strategy.
    
    1. Identify the candidate with the lowest predicted mean (mu_min) as an aggressive exploitation anchor.
    2. Compute the acquisition score as the product of sigma and the normalized distance from this anchor in prediction space.
    
    This geometric approach prioritizes high-uncertainty regions that are also far from the current best prediction,
    balancing exploration and exploitation.
    """
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Step 1: Find the anchor - candidate with lowest predicted mean
    mu_min = np.min(mu)
    
    # Step 2: Compute normalized potential based on prediction quality
    # We want to favor points with low mu (good predictions) and high sigma (uncertainty).
    # The "distance" from the worst prediction (mu_max) indicates how good the prediction is.
    mu_max = np.max(mu)
    mu_min = np.min(mu)
    mu_range = mu_max - mu_min
    
    if mu_range < 1e-9:
        # All predictions are essentially the same, rely on uncertainty
        normalized_potential = np.ones_like(mu)
    else:
        # Normalize so that mu_min gets 1.0 and mu_max gets 0.0
        # This inverts the previous "distance from best" to "closeness to best" relative to the range
        normalized_potential = (mu_max - mu) / mu_range
        
    # Step 3: Compute acquisition score as product of sigma and normalized potential
    # This rewards points that have both high uncertainty AND good (low) predicted means
    score = sigma_safe * normalized_potential
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
