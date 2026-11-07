"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0 - 30.0  # 30s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, lowest observed value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), finite scores (higher = evaluate next)
    """
    # Time guard: if we're running out of time, return a safe fallback
    if _time_left() < 0.1:
        return np.zeros_like(mu)

    # Guard against edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Hybrid acquisition: combine probability of improvement with expected improvement
    # to balance exploration (uncertainty) and exploitation (expected gain)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    z_ei_clipped = np.clip(z_ei, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_ei_clipped) + sigma_safe * norm.pdf(z_ei_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Probability of improvement with small epsilon
    epsilon = 0.01
    z_pi = (f_best - epsilon - mu) / sigma_safe
    z_pi_clipped = np.clip(z_pi, -10, 10)
    prob_improve = norm.cdf(z_pi_clipped)
    prob_improve = np.maximum(prob_improve, 0.0)
    
    # Normalize EI and PI to [0, 1] for combining
    ei_max = np.max(ei)
    pi_max = np.max(prob_improve)
    
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    if pi_max > 1e-12:
        pi_norm = prob_improve / pi_max
    else:
        pi_norm = np.zeros_like(prob_improve)
    
    # Weighted combination: 60% EI, 40% PI
    alpha = 0.6
    combined = alpha * ei_norm + (1 - alpha) * pi_norm
    
    # Cost-normalized acquisition
    scores = combined / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
