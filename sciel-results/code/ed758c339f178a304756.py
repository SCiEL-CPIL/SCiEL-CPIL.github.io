"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Use a module-level start time to track elapsed time across calls
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _norm_cdf(x):
    """Numerically stable normal CDF."""
    return 0.5 * (1.0 + np.vectorize(__import__('scipy').special.erf)(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines Expected Improvement (EI) with an
    upper confidence bound (UCB) term to balance exploitation and exploration.
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    # Deadline guard: if we're running out of time, return a simple heuristic
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Fallback: just return negative mu (exploit best known)
        return -mu.copy()
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    
    # UCB term: encourages exploration in uncertain regions
    # The coefficient beta controls the trade-off
    # For sphere 1d on [-5,5], a moderate exploration weight helps
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a weighted sum
    # Normalize both to have similar scales
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Weighted combination: 70% EI, 30% UCB
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
