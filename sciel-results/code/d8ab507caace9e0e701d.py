"""Cost-aware acquisition for Bayesian optimization on Styblinski-Tang 2D."""
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI, UCB, and PI.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition values (higher = evaluate next)
    """
    # Deadline guard: we don't have a wall-clock reference here, but we keep computation light.
    # The function is called per iteration; we ensure it's fast.
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Cost normalization: use the median cost to scale, or just divide by cost
    # Since cost is 1 + mean(x_scaled), it ranges from 1 to 2.
    # We want to penalize high-cost evaluations.
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Upper Confidence Bound (UCB) - for minimization, we want low mu + exploration
    # UCB for minimization: mu - kappa * sigma
    # But as an acquisition (higher is better), we use: -mu + kappa * sigma
    kappa = 1.0
    ucb = -mu + kappa * sigma_safe
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Blend: weight EI heavily, add some UCB exploration
    # Use adaptive weights based on how much we've explored
    # Simple fixed blend: 0.6 * EI + 0.3 * UCB + 0.1 * PI
    # Normalize each component to have similar scale before blending
    
    # Normalize EI and UCB to [0, 1] range approximately
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_range > 1e-10 else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # PI is already in [0, 1]
    
    # Blend
    score = 0.6 * ei_norm + 0.3 * ucb_norm + 0.1 * pi
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    # Use a softer penalty: divide by (cost)^alpha with alpha < 1
    alpha = 0.5
    score = score / (np.maximum(cost, 1e-10) ** alpha)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
