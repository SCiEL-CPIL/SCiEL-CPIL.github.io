"""Cost-aware acquisition function for Bayesian optimization on 3-hump camel 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = more desirable)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple, fast fallback
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB-style uncertainty bonus
    # beta controls exploration vs exploitation
    beta = 0.5
    ucb = mu - beta * sigma  # For minimization, we prefer low mu and high sigma
    
    # Combine EI and UCB signals
    # Normalize both to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_min = np.min(ucb)
    ucb_range = np.max(ucb) - ucb_min + 1e-9
    
    ei_norm = ei / (ei_max + 1e-9)
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Weighted combination: 70% EI, 30% UCB
    combined = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Cost-aware normalization
    # We want to maximize improvement per unit cost
    # Use a smooth cost penalty: divide by (1 + alpha * cost)
    alpha = 0.5
    cost_penalty = 1.0 + alpha * cost
    scores = combined / cost_penalty
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma / (np.max(sigma) + 1e-9)
    scores += exploration_bonus / cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
