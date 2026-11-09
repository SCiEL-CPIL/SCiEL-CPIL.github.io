"""Rewritten program with a stronger cost-aware acquisition for rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: keep a safety margin below the 300 s wall-clock limit.
_START_TIME = time.time()
_DEADLINE = 280.0  # seconds


def _check_deadline():
    """Return True if we are close to the wall-clock deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def _expected_improvement(mu, sigma, f_best):
    """Standard EI with a small floor on sigma to avoid division by zero."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _ucb(mu, sigma, f_best, beta):
    """Upper confidence bound (maximization form)."""
    return mu + beta * sigma


def _pi(mu, sigma, f_best):
    """Probability of improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
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

    # Compute multiple acquisition components
    ei = _expected_improvement(mu, sigma, f_best)
    pi = _pi(mu, sigma, f_best)

    # Adaptive beta for UCB: high early to escape Rastrigin's many local
    # minima, low late to polish toward the global minimum (0).
    # progress in [0, 1]: 0 when f_best is large, 1 when f_best ~ 0.
    progress = float(np.clip(1.0 - f_best / 10.0, 0.0, 1.0))
    beta = 2.5 - 1.7 * progress  # 2.5 -> 0.8

    ucb = _ucb(mu, sigma, f_best, beta)

    # Combine EI, UCB and a fading PI term with cost normalization.
    # Normalize each component by its max to make them comparable.
    ei_max = np.max(ei)
    if ei_max <= 0:
        ei_max = 1.0
    ucb_max = np.max(ucb)
    if ucb_max <= 0:
        ucb_max = 1.0
    pi_max = np.max(pi)
    if pi_max <= 0:
        pi_max = 1.0

    ei_norm = ei / ei_max
    ucb_norm = ucb / ucb_max
    pi_norm = pi / pi_max

    # Blend: exploitation (EI) grows over time, exploration (UCB, PI) fades.
    # Early: more exploration to find promising basins; late: exploit.
    w_ei = 0.5 + 0.3 * progress     # 0.5 -> 0.8
    w_ucb = 0.4 - 0.2 * progress    # 0.4 -> 0.2
    w_pi = 0.3 - 0.3 * progress     # 0.3 -> 0.0 (fades out late)

    combined = w_ei * ei_norm + w_ucb * ucb_norm + w_pi * pi_norm

    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    scores = combined / cost_safe

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
