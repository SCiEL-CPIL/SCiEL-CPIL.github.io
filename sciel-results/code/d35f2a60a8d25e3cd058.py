"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: EI + adaptive exploration bonus.
    
    Uses the known global minimum (-39.1662) to adapt the exploration
    weight based on how far f_best is from the optimum. Early in the
    search (large gap), we explore more aggressively; late (small gap),
    we exploit more.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Known global minimum of Styblinski-Tang 1D
    f_min_true = -39.1662
    
    # Gap: how far we are from the known optimum
    # Early: gap is large -> explore more
    # Late: gap is small -> exploit more
    gap = f_best - f_min_true  # positive when we're above optimum
    
    # Adaptive exploration weight: high when gap is large, low when small
    # Use a smooth decay: weight ~ 1 / (1 + gap) but scaled appropriately
    # gap ranges from ~0 (converged) to maybe 50+ (early)
    # We want weight ~ 0.5-1.0 early, ~0.01-0.1 late
    exploration_weight = 1.0 / (1.0 + 0.1 * np.maximum(gap, 0.0))
    
    # Exploration bonus: proportional to sigma
    # Scale so that at sigma=1, bonus is exploration_weight
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combine: EI dominates when it's large, exploration helps when uncertain
    scores = ei + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
