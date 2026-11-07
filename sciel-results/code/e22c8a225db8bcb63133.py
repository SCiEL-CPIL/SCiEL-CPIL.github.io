"""Rewritten program: Cost-aware acquisition with refined hybrid EI/UCB for Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 290.0  # 10s safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and UCB.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate, in [1, 2].
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Time guard: if we are close to the limit, return a simple fallback
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-8)
        return np.where(np.isfinite(ei / cost_safe), ei / cost_safe, 0.0)

    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB term: -mu + beta * sigma (for minimization)
    # We use a fixed beta for stability, as we don't have explicit step count.
    # A value of 0.5 provides a good balance between exploitation and exploration.
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB.
    # Instead of normalizing separately, we blend them directly.
    # EI is in objective units. UCB is in objective units.
    # To balance them, we can weight them.
    # Let's try a weighted sum: 0.7 * EI + 0.3 * UCB
    # Or, as suggested by findings, a hybrid with exploration bonus.
    # Score_raw = EI + beta_explore * sigma
    # This is equivalent to EI + part of UCB.
    
    # Let's use a direct cost-adjusted hybrid:
    # Score = (EI + gamma * sigma) / Cost
    # gamma controls exploration.
    gamma = 0.5
    
    scores_raw = ei + gamma * sigma_safe
    
    # Divide by cost to make it cost-aware
    cost_safe = np.maximum(cost, 1e-8)
    scores = scores_raw / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
