# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a UCB-style exploration bonus.
    
    The EI term exploits regions likely to beat f_best.
    The UCB term (scaled by confidence) explores regions with high posterior uncertainty,
    helping escape local minima in the multi-modal Hartmann-3D landscape.
    
    We use a reduced UCB weight (0.03) compared to typical values to allow more exploitation
    as the search converges, while still maintaining enough exploration to find global minima.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (standard formulation for minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration bonus: weight * sigma
    # Using a smaller weight (0.03) to balance exploitation and exploration
    ucb_weight = 0.03
    ucb_bonus = ucb_weight * sigma_safe
    
    # Combined score: EI + UCB bonus
    # Higher values indicate better candidates
    scores = ei + ucb_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
