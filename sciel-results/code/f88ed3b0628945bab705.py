"""Rewritten program using hybrid acquisition: UCB with adaptive beta and local refinement via PI."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Combines UCB (exploration) with a probability-of-improvement term (exploitation).
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Normalize f_best: starts around 25 (5^2), approaches 0
    f_norm = np.clip(f_best / 25.0, 0, 1)
    
    # Adaptive beta for UCB: high exploration early, low late
    # Use a smooth decay: beta goes from ~2.5 to ~0.2
    beta = 0.2 + 2.3 * (f_norm ** 1.3)
    
    # UCB component: -mu + beta * sigma
    ucb_component = -mu + beta * sigma
    
    # PI component: probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    pi_component = norm.cdf(z)
    
    # Weight between UCB and PI:
    # Early on (high f_norm): more UCB (exploration)
    # Late (low f_norm): more PI (exploitation)
    # w_ucb goes from 1.0 to 0.3, w_pi from 0 to 0.7
    w_ucb = 0.3 + 0.7 * f_norm
    w_pi = 1.0 - w_ucb
    
    # Combine: higher score means evaluate next
    # Scale PI to be comparable to UCB: PI is in [0,1], UCB can be large
    # Multiply PI by a scale factor related to the range of mu
    mu_range = np.max(mu) - np.min(mu) if np.max(mu) > np.min(mu) else 1.0
    pi_scale = max(mu_range, 1.0)
    
    score = w_ucb * ucb_component + w_pi * pi_scale * pi_component
    
    # Small penalty for very high sigma to prevent pure random exploration
    sigma_penalty = 0.005 * np.maximum(sigma - 1.5, 0) ** 2
    score = score - sigma_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
