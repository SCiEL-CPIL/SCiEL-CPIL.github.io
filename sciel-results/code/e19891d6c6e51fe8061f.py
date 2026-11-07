"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PoI) with a small exploration bonus
    # to encourage coverage in multi-modal landscapes.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # z-score: how far mu is below f_best in units of sigma
    z = (f_best - mu) / sigma_safe
    
    # PoI: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Add a small exploration term based on sigma to avoid over-exploitation
    # This helps in multi-modal landscapes like Branin
    # The exploration term is scaled so it doesn't dominate the PoI signal
    # Use a small weight to keep PoI as the primary driver
    exploration_weight = 0.05
    # Normalize sigma for the exploration term
    sigma_norm = sigma / (np.max(sigma) + eps)
    exploration = exploration_weight * sigma_norm
    
    # Combine: PoI dominates, exploration provides mild diversity
    scores = poi + exploration
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
