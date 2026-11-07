"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid UCB+EI acquisition function for Bayesian optimization.
    
    Uses adaptive weighting to blend UCB (exploration) and EI (exploitation)
    based on the relative scale of uncertainty vs. expected improvement.
    Cost normalization prefers cheaper evaluations.
    
    Parameters:
    - mu: array of shape (2000,) with predicted means
    - sigma: array of shape (2000,) with predicted standard deviations
    - f_best: float, the lowest observed value
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization: f_best - mu + beta * sigma
    # This encourages both low predicted value and high uncertainty
    beta = 1.5
    ucb = (f_best - mu) + beta * sigma_safe
    ucb = np.maximum(ucb, 0.0)
    
    # Adaptive weighting: when EI is large relative to sigma, favor EI (exploitation)
    # When sigma is large relative to EI, favor UCB (exploration)
    # Use a smooth blend based on the ratio
    scale = np.maximum(sigma_safe, 1e-9)
    ei_ratio = ei / scale
    # Weight for EI: higher when EI is significant relative to uncertainty
    w_ei = ei_ratio / (ei_ratio + 1.0)
    w_ucb = 1.0 - w_ei
    
    # Blend EI and UCB
    blended = w_ei * ei + w_ucb * ucb
    
    # Cost-aware normalization: prefer cheaper evaluations
    # Use a soft penalty that's not too restrictive
    scores = blended / np.sqrt(cost_safe)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
