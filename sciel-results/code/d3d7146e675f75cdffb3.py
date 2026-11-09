"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
_DEADLINE_START = time.time()
_MAX_TIME_PER_CALL = 1.5  # seconds per acquisition call (safety margin)


def _safe_ei(mu, sigma, f_best):
    """Standard EI for minimization with safe handling of edge cases."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = np.zeros_like(mu)
    mask = z > -1e-7
    ei[mask] = ((f_best - mu)[mask] * norm.cdf(z[mask]) + sigma_safe[mask] * norm.pdf(z[mask]))
    return ei


def _ucb_minimization(mu, sigma, kappa=0.5):
    """UCB for minimization: lower is better, so we want low mu and high sigma."""
    return -(mu + kappa * sigma)


def _pi_cost_adjusted(mu, sigma, f_best, cost, exploration_weight=0.1):
    """Probability of improvement, cost-adjusted, with exploration bonus."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    # Add a small exploration term based on sigma relative to scale
    scale = max(abs(f_best), 1e-6)
    exploration = exploration_weight * sigma_safe / scale
    return (pi + exploration) / np.maximum(cost, 1e-6)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with cost adjustment.
    """
    call_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Safety: if we're running out of time, return a simple heuristic
    elapsed_global = time.time() - _DEADLINE_START
    if elapsed_global > 315:
        score = -mu / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Compute core acquisition signals
    ei = _safe_ei(mu, sigma, f_best)
    ucb = _ucb_minimization(mu, sigma, kappa=0.5)
    pi_cost = _pi_cost_adjusted(mu, sigma, f_best, cost, exploration_weight=0.1)
    
    # Normalize each signal to [0, 1] range for fair combination
    def normalize(x):
        x = np.where(np.isfinite(x), x, 0.0)
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < 1e-12:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    ei_norm = normalize(ei / cost)
    ucb_norm = normalize(ucb)
    pi_norm = normalize(pi_cost)
    
    # Weighted combination: favor EI (exploitation) with UCB (exploration)
    # and PI (probabilistic improvement with cost awareness)
    combined = 0.55 * ei_norm + 0.25 * ucb_norm + 0.20 * pi_norm
    
    # Cost adjustment: divide by cost to penalize expensive evaluations
    score = combined / cost
    
    # Check time budget
    if time.time() - call_start > _MAX_TIME_PER_CALL:
        # Fallback to simpler computation
        score = (ei / cost)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
