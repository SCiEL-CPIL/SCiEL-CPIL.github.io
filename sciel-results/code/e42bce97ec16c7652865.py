"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf


def _norm_cdf(x):
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Combines Expected Improvement (EI) with a cost-aware penalty.
    EI naturally balances exploration and exploitation, and the cost
    penalty discourages expensive evaluations.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Safety deadline guard: leave margin before 330s wall-clock
    DEADLINE = 320.0  # seconds
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Expected Improvement (EI) for minimization ---
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Compute the standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # --- Cost-aware penalty ---
    # Penalize expensive evaluations by scaling EI inversely with cost.
    # Normalize cost to [0, 1] range for consistent scaling.
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
    cost_norm = (cost_safe - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Cost multiplier: cheaper points get higher multiplier
    # Use a moderate penalty to balance exploration and cost
    cost_multiplier = 1.0 / (1.0 + 2.0 * cost_norm)  # ranges from 1.0 (cheapest) to 1/3 (most expensive)
    
    # Final score: EI scaled by cost multiplier
    scores = ei * cost_multiplier
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
