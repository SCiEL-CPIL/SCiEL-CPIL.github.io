"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """Compute expected improvement with numerical stability."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use stable computation for large |z|
    ei = np.zeros_like(mu)
    mask = sigma > 1e-9
    if np.any(mask):
        z_m = z[mask]
        ei_m = (f_best[mask] - mu[mask]) * norm.cdf(z_m) + sigma[mask] * norm.pdf(z_m)
        ei[mask] = ei_m
    return ei


def _ucb_score(mu, sigma, beta):
    """Compute UCB score (higher = more promising)."""
    return -mu + beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI and UCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean
    sigma : array of shape (2000,) - GP std deviation
    f_best : float - best observed value
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute EI (expected improvement, higher = better)
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Compute UCB score (higher = more promising for exploration)
    # Use a beta that balances exploration/exploitation
    # Scale beta by the range of sigma to adapt to uncertainty level
    sigma_range = np.max(sigma) - np.min(sigma) + 1e-9
    beta = 1.5 * sigma_range
    
    ucb = _ucb_score(mu, sigma, beta)
    
    # Normalize both to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: weight EI more heavily but include UCB for exploration
    # The blend ratio can be tuned; 0.7 EI + 0.3 UCB works well
    w_ei = 0.7
    w_ucb = 0.3
    blended = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Cost-normalize: divide by cost to prefer cheaper evaluations
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
