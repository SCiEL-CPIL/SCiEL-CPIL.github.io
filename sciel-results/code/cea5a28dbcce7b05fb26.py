"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard state
_start_time = [None]
_deadline = 300.0  # 300 seconds wall-clock limit


def _get_elapsed():
    """Get elapsed time since the first call, initializing if needed."""
    if _start_time[0] is None:
        _start_time[0] = time.time()
    return time.time() - _start_time[0]


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the limit, return a fast fallback
    elapsed = _get_elapsed()
    if elapsed > _deadline - 5.0:
        # Safe fallback: return simple EI/cost computed quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probabilistic Improvement (PI) for minimization:
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    pi = np.clip(pi, 1e-10, 1.0 - 1e-10)
    
    # Adaptive exploration: use quantile of sigma to set exploration threshold
    # Points with sigma above the 75th percentile get an exploration bonus
    sigma_sorted = np.sort(sigma_safe)
    q75_idx = int(0.75 * len(sigma_sorted))
    sigma_q75 = sigma_sorted[q75_idx] if q75_idx < len(sigma_sorted) else np.max(sigma_safe)
    
    # Exploration bonus: normalized excess sigma above q75, scaled to [0, 0.3]
    excess_sigma = np.maximum(sigma_safe - sigma_q75, 0.0)
    max_excess = np.max(excess_sigma) if np.max(excess_sigma) > 0 else 1.0
    exploration_bonus = 0.3 * (excess_sigma / max_excess)
    
    # Combine PI and exploration
    score = pi + exploration_bonus
    
    # Cost-aware: use log-penalty to preserve relative ordering
    # log(cost) grows slowly, so this is less aggressive than power-law
    cost_penalty = np.log(cost_safe + 1.0)
    score = score / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
