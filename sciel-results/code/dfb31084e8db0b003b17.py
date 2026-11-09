"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid cost-aware acquisition: blends Probability of Improvement (PI)
    with a UCB exploration term, adaptively weighted by progress.
    
    PI provides bounded, sigmoidal exploitation (avoids unbounded EI in
    high-uncertainty regions). A scaled UCB term adds structured exploration.
    The blend shifts toward exploitation as f_best approaches 0.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher when mu is low and sigma is high (good for exploration)
    # Use a moderate beta; the adaptive weight controls overall exploration
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize UCB to [0, 1] range to be comparable with PI (which is in [0, 1])
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Adaptive weight: shift from exploration to exploitation as f_best -> 0
    # Griewank 2D values are in [0, 1], so f_best_norm in [0, 1]
    f_best_norm = np.clip(f_best, 0.0, 1.0)
    
    # weight_pi goes from 0.4 (early, more exploration) to 0.85 (late, more exploitation)
    weight_pi = 0.4 + 0.45 * (1.0 - f_best_norm)
    weight_ucb = 1.0 - weight_pi
    
    # Combined score
    combined = weight_pi * pi + weight_ucb * ucb_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
