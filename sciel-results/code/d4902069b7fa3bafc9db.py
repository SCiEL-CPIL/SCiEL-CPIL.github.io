"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses a hybrid acquisition combining expected improvement with a mild
exploration bonus, plus a small exploitation boost near the best prediction.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Safety: return EI as fallback
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration: use f_best as a proxy for progress
    # When f_best is still high (far from optimum ~-39.17), explore more
    # When f_best is close to optimum, exploit more
    # Scale: f_best ranges from ~+50 (random) to ~-39 (near optimum)
    # Map to exploration weight: high f_best -> more exploration
    # Use a smooth transition centered around f_best = -20
    progress = 1.0 / (1.0 + np.exp((f_best + 20.0) / 10.0))
    # progress ~0 when f_best >> -20 (early, explore), ~1 when f_best << -20 (late, exploit)
    # Exploration weight: 0.4 early, 0.15 late
    w_explore = 0.15 + 0.25 * (1.0 - progress)
    
    # Exploration term: sigma-based (uncertainty)
    # Scale sigma to be comparable to EI magnitudes
    # EI is typically O(1) to O(10) for this problem
    explore_score = sigma * w_explore
    
    # Combine: EI + exploration bonus
    scores = ei + explore_score
    
    # Additional exploitation boost: strong bonus for points where mu < f_best
    # This helps the GP refine promising regions
    mu_diff = f_best - mu  # positive when mu < f_best
    # Use a soft boost that's significant when mu is meaningfully better
    # Scale by 1/sigma to make it relative to uncertainty
    boost = np.maximum(mu_diff, 0.0) * (1.0 / (sigma + 1e-9))
    # Cap the boost to avoid dominating
    boost = np.minimum(boost, 2.0) * 0.1 * progress
    scores += boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
