"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses an adaptive blend of Expected Improvement (EI) and Upper Confidence Bound (UCB).
The exploration weight decays as the best observed value approaches the known global minimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: The evaluator kills at 330s. We track time to ensure we return safely.
_START_TIME = time.monotonic()
_TIME_BUDGET = 300.0  # Seconds
_MARGIN = 10.0        # Safety margin

# Known global minimum of Styblinski-Tang 1D
GLOBAL_MIN = -39.1662
# Approximate initial range for progress calculation.
# Random points on [-5, 5] for ST1D can yield values around 50-100.
# We use a span of 150 to cover the likely initial range to the minimum.
PROGRESS_SPAN = 150.0


def _probability_of_improvement(mu, sigma, f_best):
    """
    Calculate Probability of Improvement for minimization.
    POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    return np.clip(poi, 0.0, 1.0)


def _poi_weighted_score(mu, sigma, f_best, beta=1.0):
    """
    Calculate a POI-based acquisition score that balances probability of improvement
    with expected gain magnitude.
    
    Score = POI * (f_best - mu + beta * sigma)
    This prioritizes points with high probability of being better than f_best,
    weighted by the expected improvement magnitude.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Expected gain: how much better we expect to be
    # Use (f_best - mu) as the expected gain, plus a small exploration bonus
    expected_gain = (f_best - mu) + beta * sigma_safe
    
    # Ensure non-negative scores
    score = poi * np.maximum(expected_gain, 0.0)
    
    return score


def _min_max_normalize(x):
    """
    Min-max normalize an array to [0, 1].
    """
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / x_range


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
    # Deadline guard: If we are close to the time limit, return a safe, fast score.
    # We use a simple exploit-only score (negative mean) to avoid expensive calculations.
    if time.monotonic() - _START_TIME > _TIME_BUDGET - _MARGIN:
        # Return -mu as a fast fallback (lower mu is better, so higher score is -mu)
        # Ensure finiteness
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # 1. Calculate Progress
    # progress = 0 when f_best is far from global min (early stage)
    # progress = 1 when f_best is close to global min (late stage)
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # 2. Determine exploration parameter beta
    # Early on (progress ~ 0), use higher beta for more exploration.
    # Late on (progress ~ 1), use lower beta for more exploitation.
    # beta controls the uncertainty bonus in the POI-weighted score.
    beta = 0.5 + 1.5 * (1.0 - progress)
    
    # 3. Calculate POI-weighted score
    # This naturally balances exploration (via POI) and exploitation (via expected gain)
    scores = _poi_weighted_score(mu, sigma, f_best, beta=beta)
    
    # 4. Add a small UCB-like term for additional exploration in early stages
    # This helps escape local optima when progress is low
    if progress < 0.5:
        # Add exploration bonus scaled by (1 - progress)
        ucb_bonus = (1.0 - progress) * 0.5 * sigma
        scores = scores + ucb_bonus

    # 6. Ensure Finite Output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
