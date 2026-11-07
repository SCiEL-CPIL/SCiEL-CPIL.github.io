"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 50s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Time guard: if we're close to deadline, return a fast fallback
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-10)
        z = (f_best - np.asarray(mu, dtype=np.float64)) / sigma_safe
        z = np.clip(z, -40.0, 40.0)
        ei = (f_best - np.asarray(mu, dtype=np.float64)) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return np.where(np.isfinite(score), score, 0.0)
    
    # Coerce inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Term 1: Expected Improvement (minimisation) ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -40.0, 40.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Term 2: Posterior uncertainty, scaled to be comparable to EI ---
    # Use f_best as a natural scale; when f_best is near 0, use 1.0
    scale = max(abs(f_best), 1e-3)
    sigma_term = sigma_safe / scale
    
    # --- Term 3: Exploitation signal -- reward regions where GP mean is low ---
    exploit_term = -mu / scale
    
    # Cost normalisation: cost is in [1, 2] for griewank 2d
    inv_cost = 1.0 / np.maximum(cost, 1e-9)
    
    # Blend: EI dominates, with moderate exploration and exploitation tilt
    w_ei = 1.0
    w_sigma = 0.35
    w_exploit = 0.5
    
    score = (w_ei * ei + w_sigma * sigma_term + w_exploit * exploit_term) * inv_cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
