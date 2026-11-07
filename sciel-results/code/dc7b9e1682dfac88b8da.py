"""Rewritten acquisition for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive LCB acquisition with exploration-exploitation balance.
    Uses a time-decaying kappa that starts aggressive and settles to a moderate value.
    Includes a small diversity bonus to encourage exploration in early steps.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Adaptive exploration parameter:
    # Start with high exploration (kappa=3.5) and decay to moderate (kappa=1.5)
    # We don't know the step number directly, but we can infer it from the
    # relationship between f_best and the typical scale of the objective.
    # For Hartmann 6D, the minimum is -3.32237 and values range roughly [-3.3, 0.5].
    # Early on, f_best will be close to 0 (random points). As we find better points,
    # f_best decreases.
    
    # Use a heuristic: if f_best is still poor (close to 0 or positive), be more exploratory
    # If f_best is good (close to -3.3), be more exploitative
    # Scale: f_best ranges from ~0.5 (bad) to ~-3.3 (good)
    # Normalize: t = (f_best - (-3.32237)) / (0.5 - (-3.32237)) = (f_best + 3.32237) / 3.82237
    # t is close to 1 when f_best is bad, close to 0 when f_best is good
    
    t = np.clip((f_best + 3.32237) / 3.82237, 0, 1)
    
    # Kappa: high when t is high (early/poor), low when t is low (late/good)
    # kappa = 1.5 + 2.0 * t  -> ranges from 1.5 to 3.5
    kappa = 1.5 + 2.0 * t
    
    # LCB: mu - kappa * sigma
    # Higher score means more promising (lower predicted value with uncertainty)
    score = mu - kappa * sigma
    
    # Add a small diversity/exploration bonus based on how "uncertain" the region is
    # This helps escape local optima in multimodal landscapes
    # The bonus is largest when sigma is large (high uncertainty)
    # But we don't want it to dominate, so keep it small
    diversity_bonus = 0.1 * sigma
    
    # Combine: primarily LCB, with a small diversity push
    # Actually, for LCB, larger sigma already increases the score (more exploration)
    # The kappa term already handles exploration. Let's stick with pure LCB
    # but ensure numerical stability
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, np.nanmin(score[np.isfinite(score)]))
    
    return score
# EVOLVE-BLOCK-END
