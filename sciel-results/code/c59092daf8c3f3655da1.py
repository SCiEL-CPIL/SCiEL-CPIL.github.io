"""Acquisition function for Bayesian optimization of Beale 2D.

Uses Expected Improvement with an adaptive uncertainty bonus.
The EI term drives exploitation toward the predicted minimum,
while a small sigma-weighted exploration term encourages
visiting uncertain regions, especially early in the search.
The exploration weight decays as the optimization progresses,
inferred from the gap between f_best and the best predicted mean.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Where sigma is ~0, EI should be max(0, f_best - mu)
    ei = np.where(sigma < eps, np.maximum(f_best - mu, 0.0), ei)
    return np.where(np.isfinite(ei), ei, 0.0)


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Numerical safety
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Core EI score
    ei = _expected_improvement(mu, sigma, f_best)

    # Progress proxy: how close is the best predicted mean to f_best?
    # Early: min(mu) is far below f_best → large gap → more exploration
    # Late: min(mu) close to f_best → small gap → less exploration
    mu_min = float(np.min(mu))
    gap = f_best - mu_min  # positive if predictions suggest improvement possible

    # Sigma scale: typical uncertainty level
    sigma_med = float(np.median(sigma))
    if sigma_med < 1e-10:
        sigma_med = 1e-10

    # Adaptive exploration weight:
    # - Large gap (early) → explore more
    # - Small sigma (late, well-explored) → explore less
    # - Cap the weight to keep EI dominant
    # Use a smooth decay: weight ~ gap / (gap + sigma_med * scale)
    # This gives weight ~1 when gap >> sigma_med, ~0.5 when gap ~ sigma_med
    exploration_scale = 1.0  # tune: how quickly exploration fades
    gap_norm = gap / (abs(gap) + exploration_scale * sigma_med + 1e-12)
    # Clamp to [0, 1]
    gap_norm = float(np.clip(gap_norm, 0.0, 1.0))

    # Exploration bonus: proportional to sigma, scaled by progress
    # Small coefficient so EI dominates
    explore_coeff = 0.15 * gap_norm
    exploration_bonus = explore_coeff * sigma

    # Combine: EI + exploration bonus
    score = ei + exploration_bonus

    # Final safety: ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
