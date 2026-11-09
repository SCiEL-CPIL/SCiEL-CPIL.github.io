"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Probability of Improvement (POI) with a small exploration bonus.
    # POI is more conservative than EI in high-dimensional multimodal landscapes,
    # reducing premature convergence and making better use of the tight cost budget.
    #
    # P(y < f_best) = Phi((f_best - mu) / sigma)
    # We add a small penalty for high cost and a tiny exploration term for high sigma
    # to avoid completely ignoring uncertain regions.
    #
    # The score is higher = better (evaluate next).
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Cost penalty: prefer cheaper points. cost is in [1, 2] (1 + mean(x)).
    # We use a multiplicative factor that slightly penalizes expensive points.
    # A gentle penalty: 1 / (1 + 0.1 * (cost - 1)) 
    # This gives ~0.909 for cost=2 and 1.0 for cost=1.
    cost_penalty = 1.0 / (1.0 + 0.1 * (cost - 1.0))
    
    # Small exploration bonus for high uncertainty, scaled to be minor
    # so that POI dominance is preserved.
    exploration_bonus = 0.05 * (sigma_safe / (sigma_safe + 0.05))
    
    score = cost_penalty * (poi + exploration_bonus)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
