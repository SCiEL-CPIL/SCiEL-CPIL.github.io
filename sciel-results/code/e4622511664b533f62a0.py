"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_DEADLINE_MARGIN = 10.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with adaptive cost-awareness.
    Uses a hybrid approach: pure EI for early iterations, then
    cost-normalized EI with adaptive beta for later iterations.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _DEADLINE_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are finite and properly shaped
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-6, 1e3)
    
    # Compute Expected Improvement (for minimization)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle the case where f_best is very negative (near optimum)
    # EI should be non-negative; clip to 0
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost-awareness:
    # Use a power penalty on cost to balance exploration vs cost
    # cost is in [1, 2] typically (1 + mean(x_scaled))
    
    # Strategy: Use UCB-like exploration with cost normalization
    # The key insight: we want to find the global minimum, so we need
    # strong exploration early, then exploitation later.
    
    # Compute a normalized sigma to understand uncertainty scale
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Adaptive beta: start high for exploration, decrease over time
    # We don't know the iteration count directly, but we can use
    # the spread of sigma as a proxy: high sigma spread = early stage
    sigma_spread = sigma_max / (sigma_mean + 1e-9)
    
    # If sigma is spread out, we're in exploration phase
    # Use a UCB-style score
    if sigma_spread > 1.5:
        # Exploration phase: strong exploration
        beta = 2.5
        ucb = mu - beta * sigma
        # Score: lower ucb is better, so negate
        base_score = -ucb
    else:
        # Exploitation phase: use EI with cost awareness
        base_score = ei
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # But don't be too aggressive - use a soft penalty
    cost_safe = np.maximum(cost, 1.0)
    
    # Normalize cost to [1, 2] range for consistent scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost penalty: use a mild power to avoid over-penalizing
    # cost_penalty = cost_norm^0.5 gives a soft penalty
    cost_penalty = 1.0 + 0.5 * cost_norm
    
    scores = base_score / cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small exploration bonus based on sigma to avoid getting stuck
    # This helps escape local optima in the cosine8 landscape
    exploration_bonus = 0.1 * sigma / (sigma_max + 1e-9)
    scores = scores + exploration_bonus
    
    # Final safety: ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
