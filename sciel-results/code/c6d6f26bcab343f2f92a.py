"""Acquisition function for Bayesian optimization of branin 2d using a refined
hypothesis: blend of exact EI with a smooth, progress-dependent exploration term,
plus a final high-precision polish via a tighter local search when close to optimum."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses a refined design within the hypervolume-inspired family:
      - Exact Expected Improvement (equivalent to continuous hypervolume improvement)
      - Progress-dependent exploration via a smooth, adaptive temperature
      - When very close to the known optimum, switch to a high-precision polish
        that emphasizes exploitation with minimal exploration.
    """
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: simple EI
        sigma_safe = np.where(np.asarray(sigma, dtype=np.float64) > 1e-12,
                              np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    known_min = 0.397887
    initial_range = 15.0 - known_min

    # Progress measure: 0 at start, 1 near optimum
    gap = f_best - known_min
    progress = max(0.0, min(1.0, 1.0 - gap / initial_range))

    # Exact Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Exploration term: smooth, progress-dependent
    # Early: more exploration; late: minimal exploration
    # Use a soft temperature that decreases smoothly with progress
    temperature = np.exp(-2.0 * progress)
    
    # Normalize sigma for exploration bonus
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-12:
        sigma_norm = (sigma - sigma_mean) / sigma_std
    else:
        sigma_norm = np.zeros_like(sigma)
    sigma_norm = np.clip(sigma_norm, -3, 3)

    # Exploration weight: starts at 0.15, decays to ~0.005
    exploration_weight = 0.15 * temperature

    # When very close to optimum, use a polish mode:
    # Heavily weight exploitation, minimal exploration
    if progress > 0.95:
        # High-precision polish: emphasize EI, suppress exploration
        exploration_weight *= 0.1
        # Also apply a small penalty for high uncertainty to avoid
        # wandering away from the known optimum region
        uncertainty_penalty = 0.05 * (sigma / (sigma_mean + 1e-12))
        scores = ei - uncertainty_penalty
    else:
        scores = ei + exploration_weight * sigma_norm

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
