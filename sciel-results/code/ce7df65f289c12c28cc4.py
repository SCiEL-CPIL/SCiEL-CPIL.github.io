import numpy as np
from scipy.stats import norm
import time

# Global state for tracking elapsed time
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before the 330s kill

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for Bayesian optimization of Hartmann 6D.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Posterior mean
    sigma: np.ndarray of shape (N,) - Posterior standard deviation
    f_best: float - Best observed value so far (minimum)
    
    Returns:
    np.ndarray of shape (N,) - Acquisition scores (higher is better)
    """
    
    # Deadline guard: if we are running out of time, return a safe fallback
    if time.time() > _DEADLINE:
        return -mu + 1.0 * sigma
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei_score = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Compute UCB score
    # UCB for minimization: -mu + beta * sigma
    # Higher UCB means more promising (lower mu, higher sigma)
    
    # Adaptive Beta:
    # We want high beta when f_best is far from the global minimum (-3.32237).
    # We want low beta when f_best is close to the global minimum.
    # Let's normalize f_best.
    # Assume f_best starts around 0-1 and goes down to -3.32.
    # Let's map f_best to a "progress" metric.
    
    # Global min is -3.32237.
    # Let's assume initial f_best is around 0.0 (or maybe 1.0).
    # We can use a sigmoid or linear mapping.
    
    # Linear mapping from f_best in [-3.32, 1.0] to beta in [0.5, 2.0]
    # When f_best = 1.0 (bad), beta = 2.0 (explore)
    # When f_best = -3.32 (good), beta = 0.5 (exploit)
    
    f_min_global = -3.32237
    f_initial_guess = 1.0  # Rough guess for initial uniform random points in [0,1]^6 for Hartmann 6D
    
    # Clamp f_best to this range for stability
    f_best_clamped = np.clip(f_best, f_min_global, f_initial_guess)
    
    # t = 0 when f_best = f_initial_guess
    # t = 1 when f_best = f_min_global
    t = (f_best_clamped - f_initial_guess) / (f_min_global - f_initial_guess)
    t = np.clip(t, 0.0, 1.0)
    
    # Beta decreases from 2.0 to 0.5
    beta = 2.0 - 1.5 * t
    
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Combine scores
    # EI and UCB have different scales.
    # EI is bounded by (f_best - min(mu)) roughly.
    # UCB is bounded by (-min(mu) + beta * max(sigma)) roughly.
    # We normalize them to have similar scales before combining.
    
    # Normalize EI
    ei_min = np.min(ei_score)
    ei_max = np.max(ei_score)
    ei_range = ei_max - ei_min
    if ei_range > 1e-9:
        ei_norm = (ei_score - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei_score)
    
    # Normalize UCB
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # We want to balance exploration (UCB) and exploitation (EI).
    # Early on (high beta, high uncertainty), UCB is better.
    # Late on (low beta, low uncertainty), EI is better.
    
    # w_ucb = 0.7 - 0.4 * t  (0.7 early, 0.3 late)
    # w_ei  = 0.3 + 0.4 * t  (0.3 early, 0.7 late)
    
    w_ucb = 0.7 - 0.4 * t
    w_ei = 1.0 - w_ucb
    
    # Combine normalized scores
    scores = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure all scores are finite
    # If any NaN or Inf appear, replace with a very low score (or 0)
    # so they are not selected.
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
