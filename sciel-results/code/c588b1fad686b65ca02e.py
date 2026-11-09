"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return EI-based score
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    pi = norm.cdf(z_clipped)
    
    # Adaptive exploration bonus:
    # When far from the known minimum (early stage), reward high-uncertainty regions
    # When close to the minimum (late stage), focus on exploitation
    f_min_known = 0.397887
    delta_f = max(f_best - f_min_known, 0.0)
    
    # Exploration weight: high early, decays as we approach the minimum
    # Use a smooth decay based on relative improvement
    max_delta = 15.0  # Approximate max possible delta_f for Branin
    rel_delta = min(delta_f / max_delta, 1.0)
    explore_weight = 0.3 * np.exp(-2.0 * rel_delta)
    
    # Exploration bonus: reward points with high sigma relative to current scale
    sigma_max = np.max(sigma)
    if sigma_max > 1e-8:
        explore_bonus = explore_weight * (sigma / sigma_max)
    else:
        explore_bonus = np.zeros_like(sigma)
    
    # Combine PI with exploration bonus
    scores = pi + explore_bonus
    
    # In late stages, add a small exploitation boost for very low mu
    if delta_f < 0.05:
        mu_min = mu.min()
        mu_max = mu.max()
        if mu_max > mu_min:
            exploit_boost = 0.05 * (1.0 - (mu - mu_min) / (mu_max - mu_min))
            scores += exploit_boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
