"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Targeted Thompson Sampling variant with cost-awareness.
    Samples an offset only from values better than (lower than) f_best,
    using a half-normal distribution scaled by the median uncertainty.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Use median sigma for robustness against outliers in uncertainty
    scale = np.median(sigma)
    
    # If scale is too small, use a small floor to allow some exploration
    scale = max(scale, 1e-3)
    
    # Sample a "target" value that is better than f_best.
    # Using a half-normal distribution ensures offset <= f_best.
    # This directs the acquisition towards promising (lower) regions.
    offset = f_best - np.abs(np.random.normal(0, scale))
    
    # Score: (offset - mu) / cost
    # We want mu to be close to offset (which is < f_best).
    # Higher score means mu is lower (closer to offset) and cost is lower.
    scores = (offset - mu) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
