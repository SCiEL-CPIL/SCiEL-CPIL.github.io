"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive Expected Improvement (EI)
    with a logarithmic beta schedule for exploration/exploitation balance.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive beta using logarithmic schedule for smoother decay
    # f_best is negative (we're minimizing), so log(1 + f_best) needs care
    # Use abs to ensure positive argument, and use a multiplier for stronger early exploration
    f_best_abs = abs(f_best)
    beta = np.log(1.0 + f_best_abs) * 2.0
    # Floor at 1 to ensure at least exploitation
    beta = max(beta, 1.0)
    
    # Cost-normalized acquisition with exploration bonus
    # Higher beta means more weight on exploration (sigma term)
    scores = (ei + beta * sigma_safe) / (cost + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
