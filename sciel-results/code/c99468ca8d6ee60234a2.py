"""Cost-aware acquisition using cost-normalized Quantile of the Posterior (QoP) for rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: keep a safety margin below the 300 s wall-clock limit.
_START_TIME = time.time()
_DEADLINE = 285.0  # seconds


def _check_deadline():
    """Return True if we are close to the wall-clock deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def _qop(mu, sigma, f_best, progress):
    """
    Quantile of the Posterior (QoP) acquisition for minimization.
    
    Returns the predicted p-th quantile of f(x) ~ N(mu, sigma^2), where p is
    adapted by progress:
      - Early (low progress): p is high (e.g., 0.90) -> explore uncertain regions
      - Late (high progress): p drops toward 0.50 -> exploit near the mean
    
    The score is -QoP (higher = better for minimization), blended with a small
    PI-like bonus that reinforces improvement probability.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adapt quantile level: early = 0.90 (explore), late = 0.50 (exploit)
    # Smooth interpolation using progress
    p = 0.50 + 0.40 * (1.0 - progress) ** 1.5
    
    # z-score for the p-th quantile (lower tail for minimization)
    # We want the p-th quantile of the objective; for minimization, lower is better
    # QoP = mu + z_p * sigma, where z_p = norm.ppf(p)
    # Since p > 0.5, z_p > 0, so QoP > mu (conservative, higher predicted value)
    # For minimization, we want LOW QoP, so we minimize QoP = mu + z_p * sigma
    # Score = -QoP (maximize this)
    z_p = norm.ppf(p)
    qop_value = mu + z_p * sigma_safe
    
    # Primary score: negative QoP (we want low quantile values)
    score = -qop_value
    
    # Add a small PI bonus to reinforce regions likely to improve
    z_pi = (f_best - mu) / sigma_safe
    pi_bonus = norm.cdf(z_pi)
    
    # Blend: QoP primary, PI secondary (scaled by progress to fade late)
    w_pi = 0.15 * progress
    score = score + w_pi * pi_bonus
    
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Uses cost-normalized Quantile of the Posterior (QoP) as the primary signal,
    with an adaptive quantile level that shifts from exploration (high p) early
    to exploitation (low p) late. QoP directly targets high-confidence low-value
    regions while the cost normalization explicitly penalizes expensive evaluations.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: if we are near the wall-clock limit, return a safe
    # exploitation-focused score immediately.
    if _check_deadline():
        cost_safe = np.maximum(cost, 1e-9)
        fallback = -mu / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)

    # Progress metric: 0 when f_best is large (early), 1 when f_best ~ 0 (late)
    # Rastrigin 2d max value is around 40, so normalize by that
    progress = float(np.clip(1.0 - f_best / 40.0, 0.0, 1.0))

    # QoP acquisition with adaptive quantile
    scores = _qop(mu, sigma, f_best, progress)

    # Normalize by cost to penalize expensive evaluations
    scores = scores / cost_safe

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
