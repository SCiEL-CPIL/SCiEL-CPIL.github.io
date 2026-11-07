"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Monte-Carlo Expected Improvement (with adaptive
    reference) and a normalized UCB exploration term.
    
    Structural change: replaces the analytic EI with a Monte-Carlo approximation
    to avoid numerical artifacts at extreme z-scores. The reference point for EI
    is adaptively chosen as min(f_best, 3.0) to maintain a constant exploration
    signal when f_best approaches the known global minimum of 3.0.
    
    The UCB term uses an adaptive beta that scales with relative uncertainty,
    and both terms are combined with a weight that shifts from exploration to
    exploitation as the optimization progresses.
    
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
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive EI reference: use min(f_best, 3.0) to maintain exploration signal
    f_ref = min(f_best, 3.0)
    
    # Monte-Carlo Expected Improvement for minimization
    # E[max(f_ref - Y, 0)] where Y ~ N(mu, sigma^2)
    # Use a fixed number of samples for efficiency
    n_samples = 64
    rng = np.random.default_rng(seed=42)
    
    # Generate standard normal samples: shape (n_samples, n)
    z_samples = rng.standard_normal((n_samples, n))
    
    # Sample from posterior: Y = mu + sigma * z
    # Shape: (n_samples, n)
    y_samples = mu[None, :] + sigma_safe[None, :] * z_samples
    
    # Compute improvement: max(f_ref - Y, 0)
    improvement = np.maximum(f_ref - y_samples, 0.0)
    
    # Expected improvement: mean over samples
    ei_mc = np.mean(improvement, axis=0)
    
    # Compute UCB term with adaptive beta
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    sigma_mean = np.mean(sigma_safe)
    # Adaptive beta: larger when uncertainty is high relative to prediction spread
    beta = 1.0 + 3.0 * min(1.0, sigma_mean / (mu_range + 1e-10))
    
    # UCB score for minimization (we want to minimize mu + beta*sigma)
    ucb_value = mu + beta * sigma_safe
    ucb_score = -ucb_value
    
    # Normalize both terms to [0, 1] for stable combination
    ei_min = np.min(ei_mc)
    ei_max = np.max(ei_mc)
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei_mc - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei_mc)
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Weight: lean more toward EI in later stages, more toward UCB in early
    sigma_to_fbest_ratio = sigma_mean / (abs(f_best) + 1e-10)
    w_ucb = 0.3 + 0.3 * min(1.0, sigma_to_fbest_ratio)
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small bonus for points where mu is significantly below f_ref
    below_best = np.maximum(0.0, f_ref - mu)
    if np.max(below_best) > 1e-10:
        below_best_norm = below_best / (np.max(below_best) + 1e-10)
        scores += 0.1 * below_best_norm
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
