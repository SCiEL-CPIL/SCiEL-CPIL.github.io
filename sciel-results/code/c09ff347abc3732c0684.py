"""Rewritten cost-aware acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Leave 35s buffer before 330s kill


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    mu, sigma, cost: shape (2000,)
    f_best: scalar, lowest observed value
    Returns: shape (2000,) finite scores, higher = more promising
    """
    # Deadline guard
    if _remaining_time() < 3.0:
        # Simple fallback: EI / cost
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-10)
        return np.where(np.isfinite(score), score, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB with adaptive beta
    # Estimate how close we are to convergence
    # Use the spread of mu as a proxy for remaining uncertainty
    mu_range = np.max(mu) - np.min(mu) if len(mu) > 0 else 1.0
    if mu_range < 1e-8:
        mu_range = 1e-8
    # Normalize: if f_best is well below the range, we're converged
    conv_signal = np.clip((f_best - np.min(mu)) / (mu_range + 1e-10), 0.0, 1.0)
    # beta: higher when more uncertainty remains
    beta = 1.0 + 2.0 * (1.0 - conv_signal)
    
    ucb = mu + beta * sigma_safe
    
    # Cost-aware combinations
    # Use sqrt(cost) normalization to be less aggressive than 1/cost
    cost_norm = np.sqrt(cost_safe)
    
    ei_cost = ei / cost_norm
    ucb_cost = -ucb / cost_norm  # negate since lower ucb is better
    
    # Blend: primarily EI-based, with UCB providing exploration
    # Weight EI more when it's significant, UCB more when EI is small
    ei_mag = np.mean(ei_cost) if len(ei_cost) > 0 else 0.0
    ucb_mag = np.mean(np.abs(ucb_cost)) if len(ucb_cost) > 0 else 0.0
    
    if ei_mag < 1e-8 and ucb_mag < 1e-8:
        # Degenerate case: use UCB
        score = ucb_cost
    elif ei_mag < 1e-4:
        # EI is very small, rely more on exploration
        w_ei = 0.3
        w_ucb = 0.7
    elif ucb_mag < 1e-4:
        # UCB signal is weak, rely on exploitation
        w_ei = 0.8
        w_ucb = 0.2
    else:
        # Balanced regime
        # Scale weights by relative magnitudes
        total = ei_mag + ucb_mag
        w_ei = 0.5 * ei_mag / total + 0.3
        w_ucb = 0.5 * ucb_mag / total + 0.2
        # Normalize
        w_sum = w_ei + w_ucb
        w_ei /= w_sum
        w_ucb /= w_sum
    
    score = w_ei * ei_cost + w_ucb * ucb_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
