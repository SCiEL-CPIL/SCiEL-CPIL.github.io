"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """
    Expected Improvement: E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu, sigma^2).
    """
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Compute UCB-like score for exploration
    # For minimization, we want low values, so UCB = mu - beta * sigma
    # But for acquisition (higher is better), we want to reward low mu and high sigma
    # Let's use a hybrid approach
    
    # Soften the cost penalty: use cost^0.4 instead of cost^1.0
    c = np.maximum(cost, 1e-9)
    cost_softened = np.power(c, 0.4)
    
    # Blend EI with UCB component
    # UCB component for minimization: lower mu is better, higher sigma is better
    # We can use: -mu + beta * sigma as a score (higher is better)
    # But we need to normalize or scale appropriately
    
    # Let's compute a simple UCB score
    # For minimization, we want to find points with low predicted values and high uncertainty
    # UCB_score = -mu + beta * sigma (higher is better for acquisition)
    beta = 0.5
    ucb_score = -mu + beta * s
    
    # Blend: 70% EI + 30% UCB
    # Need to normalize both to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 0 else 1.0
    
    ei_normalized = ei / ei_max
    ucb_normalized = ucb_score / ucb_max
    
    # Ensure non-negative
    ucb_normalized = np.maximum(ucb_normalized, 0.0)
    
    # Blend
    blended = 0.7 * ei_normalized + 0.3 * ucb_normalized
    
    # Apply softened cost penalty
    score = blended / cost_softened
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we've spent too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # 20s buffer before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
