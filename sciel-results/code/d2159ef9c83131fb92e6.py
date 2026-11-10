"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for tracking elapsed time and potentially adaptive parameters
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Cost-Adjusted UCB (CA-UCB) with adaptive beta.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Safety margin: if we're getting close to the 330s limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Return a simple, fast acquisition to avoid timeout
        return -mu / (cost + 1e-9)
    
    # Adaptive beta: start high for exploration, decrease over time
    # Use a schedule that balances exploration and exploitation
    # Based on the number of calls (each call represents one iteration of the loop)
    # The loop runs while total cost < 30, starting with 5 points.
    # Each evaluation costs 1 + mean(x_scaled), so roughly 1-2 per eval.
    # So we might have ~15-25 evaluations total.
    
    # Use a logarithmic schedule for beta
    # beta(t) = beta_0 * log(1 + t / t0) / (1 + t / t0)
    # This starts at beta_0 and decays
    beta_0 = 2.0  # Initial exploration weight
    t0 = 5.0      # Time scale for decay
    
    # Use call count as a proxy for time/progress
    t = float(_call_count)
    beta = beta_0 * np.log(1.0 + t / t0) / (1.0 + t / t0)
    
    # Clamp beta to reasonable range
    beta = np.clip(beta, 0.5, 3.0)
    
    # Cost-Adjusted UCB:
    # score = (f_best - mu + beta * sigma) / cost
    # This is equivalent to maximizing expected improvement per unit cost
    # with an explicit exploration term
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute the acquisition score
    # The numerator is: f_best - mu + beta * sigma
    # This is the UCB value (we want to minimize, so higher UCB is better)
    ucb = f_best - mu + beta * sigma_safe
    
    # Divide by cost to get cost-adjusted score
    scores = ucb / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
