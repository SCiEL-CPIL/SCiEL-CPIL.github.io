"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 295.0  # 5s safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Ackley 2D minimization.
    
    Uses a hybrid of exploitation (low predicted mean) and exploration (high uncertainty),
    normalized by cost. The beta parameter balances exploration vs exploitation.
    """
    # Check deadline - if we're running out of time, return safe default
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and extreme values
    sigma_clipped = np.clip(sigma, 1e-6, None)
    
    # Cost normalization: lower cost is better
    cost_clipped = np.clip(cost, 0.1, None)
    
    # UCB: f_best - mu + beta * sigma
    # beta controls exploration-exploitation tradeoff
    # For tight budget on Ackley 2D, moderate exploration is key
    beta = 2.0
    
    # Score: negative of (predicted value - beta*uncertainty) / cost
    # We want to maximize this, so we negate the predicted regret
    # ucb_score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Add a small bonus for points that could potentially beat f_best
    # This helps escape local minima
    potential_improvement = np.maximum(f_best - mu, 0)
    
    # Combined score: balance between UCB exploration and potential improvement
    # The cost-aware aspect: divide by cost to prefer cheaper evaluations
    score = (beta * sigma_clipped + 0.5 * potential_improvement) / cost_clipped
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
