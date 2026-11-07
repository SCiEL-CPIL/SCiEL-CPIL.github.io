"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PoI) with a small exploration bonus.
    # PoI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We add a small term proportional to sigma to encourage exploration
    # in uncertain regions, which helps cover the multi-modal Branin landscape.
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Small exploration bonus: favors points with higher uncertainty
    # but scaled down so that PoI dominates
    exploration_bonus = 0.05 * (sigma / (sigma_safe + 1e-9))
    
    scores = poi + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
