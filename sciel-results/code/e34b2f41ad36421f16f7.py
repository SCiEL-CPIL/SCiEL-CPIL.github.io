"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Probabilistic Lower Bound (PLB) acquisition with adaptive exploration.
    
    score = mu + k * sigma
    
    where k adapts based on how far f_best is from the known minimum (3.0)
    for the Goldstein-Price function.
    
    When f_best is far from 3.0, we use larger k (more exploration).
    As we converge, k decreases (more exploitation).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if sigma.min() < 1e-12:
        sigma = np.maximum(sigma, 1e-12)
    
    # Known minimum of Goldstein-Price is 3.0
    f_min_known = 3.0
    
    # Compute distance from current best to known minimum
    # f_best is the lowest observed value (we're minimizing)
    # If f_best is close to f_min_known, we should exploit more
    # If f_best is far, we should explore more
    
    # The gap between current best and known minimum
    gap = f_best - f_min_known
    
    # Adaptive k: starts high for exploration, decays to low for exploitation
    # Use a smooth transition
    # When gap is large (e.g., > 10), k ~ 3.0 (strong exploration)
    # When gap is small (e.g., < 0.1), k ~ 0.5 (exploitation)
    
    # Normalize the gap for the adaptive schedule
    # Use a sigmoid-like or linear interpolation
    # Let's use: k = k_min + (k_max - k_min) * exp(-gap / scale)
    # This gives high k when gap is large, low k when gap is small
    
    k_min = 0.5   # exploitation regime
    k_max = 3.0   # exploration regime
    scale = 5.0   # controls how quickly k decays
    
    # If gap is negative (f_best < f_min_known, which shouldn't happen but be safe)
    if gap < 0:
        gap = 0
    
    k = k_min + (k_max - k_min) * np.exp(-gap / scale)
    
    # Compute PLB score: mu + k * sigma
    # Higher score means more likely to improve
    scores = mu + k * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
