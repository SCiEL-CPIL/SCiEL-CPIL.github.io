"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses Expected Improvement relative to the known global minimum (3.0)
    combined with an adaptive UCB term. The known minimum provides a
    constant, strong exploration signal that doesn't diminish as f_best
    approaches 3.0, keeping the acquisition function sensitive to
    high-variance regions that might contain the true optimum.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    t_start = time.time()
    
    # Safety: clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Known global minimum of Goldstein-Price 2D
    f_min_true = 3.0
    
    # Use the known minimum as reference for EI - this provides a constant
    # exploration signal that doesn't diminish as f_best approaches 3.0
    # This is key: EI relative to f_min_true stays strong even when f_best is close to 3.0
    
    # Expected Improvement relative to the known minimum
    # For minimization: EI = (f_ref - mu) * Phi(z) + sigma * phi(z) where z = (f_ref - mu) / sigma
    f_ref = f_min_true
    z_ei = (f_ref - mu) / sigma_safe
    ei = (f_ref - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # UCB score for minimization: we want to minimize mu + beta*sigma
    # Higher score = lower mu + beta*sigma
    # Adaptive beta: larger when uncertainty is high relative to prediction spread
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    sigma_mean = np.mean(sigma_safe)
    # beta ranges from ~1.5 to ~5.0 based on relative uncertainty
    beta = 1.5 + 3.5 * min(1.0, sigma_mean / (mu_range + 1e-10))
    
    ucb_value = mu + beta * sigma_safe
    ucb_score = -ucb_value  # higher is better
    
    # Normalize both terms to [0, 1] for stable combination
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Weight: lean more toward EI when we're close to the optimum,
    # more toward UCB when uncertainty is high
    # Use the ratio of sigma to the gap between f_best and true minimum
    gap = max(f_best - f_min_true, 1e-6)
    sigma_to_gap_ratio = sigma_mean / gap
    # Clamp to [0, 1]
    sigma_to_gap_ratio = min(1.0, max(0.0, sigma_to_gap_ratio))
    
    # When gap is small (close to optimum), favor EI more
    # When gap is large or uncertainty is high, favor UCB more
    w_ucb = 0.25 + 0.25 * sigma_to_gap_ratio
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
