"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute constants for speed
_EPS = 1e-9
_NORM_CDF = norm.cdf
_NORM_PDF = norm.pdf

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # 5s safety margin before 330s kill


def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Expected Regret per unit cost with exploration component.
    
    Uses Expected Regret (ER) which directly measures expected reduction in
    the objective, normalized by cost. Adds a structured exploration term
    that is more effective for multimodal landscapes like Rastrigin.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of GP predictions at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of GP predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline check - return safe fallback if over time
    if _check_deadline():
        return np.zeros(2000, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, _EPS)
    cost_safe = np.maximum(cost, _EPS)
    
    # Compute z-score for EI/ER calculation
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (standard)
    ei = (f_best - mu) * _NORM_CDF(z) + sigma_safe * _NORM_PDF(z)
    
    # Expected Regret: E[max(0, f(x) - f_best)] 
    # This is equivalent to EI but framed as expected regret reduction
    # ER = E[max(0, f(x) - f_best)] = same formula as EI
    er = ei
    
    # Cost-aware: ER per unit cost
    cost_aware_er = er / cost_safe
    
    # Structured exploration: use a UCB-like term with adaptive beta
    # that scales with the uncertainty relative to the current best gap
    # For multimodal landscapes, we want to explore regions with high sigma
    # but weight it by how far the mean is from f_best (exploitation pressure)
    
    # Adaptive exploration coefficient:
    # When f_best is close to 0 (good solution found), reduce exploration
    # When f_best is far from 0, increase exploration
    gap = np.maximum(f_best, _EPS)
    
    # Beta adapts: larger when we haven't found a good solution yet
    # Use a smooth transition
    beta = 0.3 * (1.0 + 1.0 / (1.0 + gap))
    
    # UCB term: mu + beta * sigma, but we want to maximize improvement
    # So the exploration bonus should favor high sigma regions
    ucb_bonus = beta * sigma_safe / cost_safe
    
    # Additional term: favor points where the GP predicts values BELOW f_best
    # but with high uncertainty (could be much better)
    # This is a "potential improvement" term
    potential = np.maximum(0, f_best - mu) / cost_safe
    potential_weight = 0.1 * sigma_safe / (sigma_safe + _EPS)
    potential_term = potential_weight * potential
    
    scores = cost_aware_er + ucb_bonus + potential_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
