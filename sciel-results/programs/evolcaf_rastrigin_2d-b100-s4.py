"""Cost-aware acquisition using cost-normalized PI with adaptive UCB for rastrigin 2d."""
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


def _pi(mu, sigma, f_best):
    """Probability of Improvement: P(f(x) < f_best) for f(x) ~ N(mu, sigma^2)."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def _ucb_score(mu, sigma, beta):
    """UCB score: mu + beta * sigma (for minimization, we maximize -mu + beta*sigma)."""
    return -mu + beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Structural change: uses cost-normalized Probability of Improvement (PI) as the
    primary signal, combined with an adaptive UCB term for exploration.
    
    PI has a fundamentally different decision boundary than ER/EI: it directly
    measures the probability that a candidate improves upon f_best, which is
    more robust to the multi-modal landscape of Rastrigin.
    
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

    # Primary signal: Probability of Improvement
    pi = _pi(mu, sigma, f_best)

    # Adaptive UCB beta: higher early for exploration, lower late for exploitation
    # Quadratic decay: starts at 1.0, ends at 0.1
    beta = 0.1 + 0.9 * (1.0 - progress) ** 2

    # UCB component for exploration
    ucb = _ucb_score(mu, sigma, beta)

    # Combine: PI as primary (normalized by cost), UCB as secondary exploration
    # The UCB term is scaled down by progress to fade out late
    w_ucb = 0.3 * (1.0 - progress)
    
    scores = pi / cost_safe + w_ucb * ucb / cost_safe

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
