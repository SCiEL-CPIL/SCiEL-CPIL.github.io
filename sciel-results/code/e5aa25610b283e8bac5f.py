"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Thompson Sampling with adaptive cost weighting.
    
    Uses Thompson Sampling to draw a sample from the posterior N(mu, sigma^2) for each candidate.
    The score is -sampled_value / (cost ** alpha), where alpha adapts based on remaining budget.
    This combines the exploration benefits of Thompson Sampling with the cost-awareness of PI/cost.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _deadline, _start_time, _call_count
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # Safety margin before 330s limit
    
    # Increment call count
    _call_count += 1
    
    # Check deadline - emergency fallback if we're running out of time
    if time.time() > _deadline:
        # Simple fallback: return uniform scores to let the loop proceed
        return np.ones(len(mu))
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero or negative values
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Draw samples from the posterior distribution
    n_samples = mu.shape[0]
    z = np.random.normal(0, 1, size=n_samples)
    
    # Sampled objective values
    y_samples = mu + sigma_safe * z
    
    # Estimate remaining budget
    # Initial 5 points: uniform in [0,1]^6, cost = 1 + mean(x) where x in [0,1]^6
    # E[mean(x)] = 0.5, so E[cost] = 1.5 per point
    # 5 initial points: ~7.5 cost
    # Each subsequent point: ~1.5 cost on average
    initial_cost_est = 7.5
    avg_cost_per_step = 1.5
    total_cost_est = initial_cost_est + _call_count * avg_cost_per_step
    remaining_budget = max(0.0, 30.0 - total_cost_est)
    
    # Adaptive cost exponent:
    # When budget is plentiful (remaining > 10), alpha = 1.0 (standard cost-awareness)
    # When budget is tight (remaining < 5), alpha = 2.0 (strong cost avoidance)
    # Linear interpolation in between
    if remaining_budget > 10.0:
        alpha = 1.0
    elif remaining_budget < 5.0:
        alpha = 2.0
    else:
        alpha = 1.0 + (10.0 - remaining_budget) / 5.0
    
    # Score = -y_samples / (cost ** alpha)
    # We want to minimize y_samples, so higher score for lower y_samples
    # Dividing by cost^alpha makes cheaper points more attractive
    scores = -y_samples / (cost_safe ** alpha)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
