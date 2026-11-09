"""Cost-aware acquisition for Levy 2D minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state for deadline guarding
_start_time = None
_DEADLINE_SECONDS = 310.0  # 20s safety margin before 330s kill

def _ensure_init():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_exceeded():
    return (time.time() - _start_time) > _DEADLINE_SECONDS


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using PoI + entropy exploration.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    _ensure_init()
    
    # Deadline guard: if we're running out of time, return a simple but safe score
    if _time_exceeded():
        # Return EI-based score without expensive operations
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei / (np.maximum(cost, 1e-9))
    
    n = len(mu)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PoI): P(f(x) < f_best)
    # This is more aggressive than EI - it rewards points with high probability of improving
    poi = norm.cdf(z)
    
    # Entropy-based exploration bonus: encourages visiting regions with high uncertainty
    # Using differential entropy of the Gaussian: 0.5 * log(2*pi*e*sigma^2)
    # This is structurally different from a linear sigma bonus
    entropy = 0.5 * np.log(2 * np.pi * np.e * sigma_safe ** 2)
    
    # Normalize entropy to [0, 1] range for stable mixing
    entropy_min = np.min(entropy)
    entropy_max = np.max(entropy)
    entropy_range = entropy_max - entropy_min + 1e-9
    entropy_norm = (entropy - entropy_min) / entropy_range
    
    # Weighted combination: PoI (exploit) + entropy (explore)
    # The weight adapts based on how close mu is to f_best
    # When mu is close to f_best, we want more exploration
    closeness = np.clip((f_best - mu) / (sigma_safe + 1e-9), 0, 10) / 10
    explore_weight = 0.3 + 0.4 * (1 - closeness)  # 0.3 to 0.7
    
    score = (1 - explore_weight) * poi + explore_weight * entropy_norm
    
    # Cost-aware: divide by cost
    score = score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
