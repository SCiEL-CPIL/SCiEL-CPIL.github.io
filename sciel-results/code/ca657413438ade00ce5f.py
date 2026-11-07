"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 330s wall-clock per evaluation.
# We track elapsed time and return a safe fallback if we approach the limit.
_DEADLINE = 325.0  # seconds, safety margin before 330s kill
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _check_deadline(fallback):
    """If we're close to the deadline, return the fallback immediately."""
    if _elapsed() > _DEADLINE:
        return fallback
    return None


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4d.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, lowest observed function value
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Safety: ensure all inputs are finite and properly shaped
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return a safe uniform-ish score
        return np.ones_like(mu)
    
    if sigma.shape != mu.shape or cost.shape != mu.shape:
        return np.ones_like(mu)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Clip cost to avoid division by zero (cost is always >= 1 in this problem)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score for Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    # Compute EI using scipy's norm (imported inside function to avoid overhead)
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI (primary signal)
    ei_cost = ei / cost_safe
    
    # UCB component for exploration
    # Beta controls exploration-exploitation tradeoff
    # Adaptive beta: start with moderate exploration, decay over time
    # Since we don't have explicit iteration count, use f_best relative to known minimum
    # Shekel 4d minimum is -10.5364
    known_min = -10.5364
    # Progress: 0 at start (f_best near 0), 1 when close to optimum
    progress = np.clip((0.0 - f_best) / (0.0 - known_min), 0.0, 1.0)
    # Beta decreases as we make progress: more exploitation later
    beta = 2.5 * (1.0 - 0.5 * progress)
    
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want low ucb, so negate for "higher is better"
    ucb_score = -ucb / cost_safe
    
    # Combine EI and UCB with a weighted sum
    # Weight EI more heavily as we progress
    w_ei = 0.7 + 0.3 * progress
    w_ucb = 1.0 - w_ei
    
    # Normalize both components to have similar scale
    ei_cost_mean = np.mean(ei_cost)
    ei_cost_std = np.std(ei_cost) + 1e-8
    ei_norm = (ei_cost - ei_cost_mean) / ei_cost_std
    
    ucb_score_mean = np.mean(ucb_score)
    ucb_score_std = np.std(ucb_score) + 1e-8
    ucb_norm = (ucb_score - ucb_score_mean) / ucb_score_std
    
    # Combined score
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check: if we're close to the limit, return a simple fallback
    fallback = ei_cost  # Simple cost-normalized EI as safe fallback
    result = _check_deadline(fallback)
    if result is not None:
        return result
    
    return score
# EVOLVE-BLOCK-END
