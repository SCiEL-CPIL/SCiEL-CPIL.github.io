import numpy as np
from scipy.stats import norm
import time

# Global state for tracking time and best result
_acq_start_time = None
_acq_best_scores = None
_ACQ_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with time guard.
    Uses UCB (Upper Confidence Bound) with cost-aware penalty.
    
    Args:
        mu: array of GP means, shape (N,)
        sigma: array of GP standard deviations, shape (N,)
        f_best: scalar, best function value observed so far
        cost: array of costs, shape (N,)
        
    Returns:
        scores: array of acquisition values, shape (N,), higher is better
    """
    global _acq_start_time, _acq_best_scores
    
    # Initialize start time on first call
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    elapsed = time.time() - _acq_start_time
    if elapsed > 330.0 - _ACQ_DEADLINE_MARGIN:
        if _acq_best_scores is not None:
            return _acq_best_scores
        # If we haven't computed anything yet, return zeros to be safe
        return np.zeros(len(mu))
    
    # Prevent division by zero in z calculation
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB for minimization: we want to minimize, so we prefer low mu + high uncertainty
    # UCB = mu - beta * sigma (lower is better)
    # But acquisition scores should be higher = better, so we negate:
    # score = -mu + beta * sigma
    
    # Adaptive beta based on number of evaluations (estimated from cost)
    # More evaluations -> less exploration needed
    total_cost = np.sum(cost)
    # beta starts high and decreases as we gather more data
    # Using a logarithmic decay: beta = beta_0 / log(1 + total_cost)
    beta_0 = 3.0
    beta = beta_0 / np.log1p(total_cost)
    
    # UCB score for minimization (higher is better)
    ucb = -mu + beta * sigma
    
    # Cost-aware penalty: penalize expensive points
    # Use exponential decay: score = ucb * exp(-k * cost)
    # But we want to balance exploration and cost
    # Alternative: score = ucb - gamma * cost
    # Let's use a multiplicative penalty with adaptive gamma
    
    # Normalize cost to [0, 1] range for consistent penalty
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_normalized = np.zeros_like(cost)
    else:
        cost_normalized = (cost - cost_min) / cost_range
    
    # Adaptive penalty strength: stronger penalty when budget is tight
    # As total_cost approaches 30, we should be more cost-sensitive
    budget_fraction = total_cost / 30.0
    gamma = 1.0 + 2.0 * budget_fraction  # gamma ranges from 1 to 3
    
    # Final score: UCB minus cost penalty
    scores = ucb - gamma * cost_normalized
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Update best scores
    _acq_best_scores = scores.copy()
    
    return scores
