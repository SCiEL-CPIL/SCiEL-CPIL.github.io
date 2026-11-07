"""Cost-aware EI with variance penalty and dynamic exploration scaling."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 300s limit + 5s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using EI with variance penalty.
    
    Combines expected improvement with a quadratic variance penalty that
    discourages high-uncertainty regions, scaled dynamically by the gap
    to the known minimum. Cost normalization favors cheaper evaluations.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety: return deterministic fallback
        sigma_safe = np.maximum(np.asarray(sigma), 1e-9)
        cost_safe = np.maximum(np.asarray(cost), 1e-9)
        z = (f_best - np.asarray(mu)) / sigma_safe
        ei = (f_best - np.asarray(mu)) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardize improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (stable component)
    pi = norm.cdf(z)
    
    # Dynamic exploration scaling based on gap to minimum
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # Variance penalty: quadratic in sigma, scaled by gap
    # Larger gap -> more exploration allowed (smaller penalty)
    # Smaller gap -> tighter penalty to focus on exploitation
    # Base penalty coefficient: 0.01 (from inspiration 1)
    # Scale: when gap is large, reduce penalty to allow more exploration
    # When gap is small, increase penalty to focus on low-uncertainty regions
    penalty_scale = 0.01 * (1.0 + 2.0 * np.exp(-gap_to_min / 0.3))
    variance_penalty = penalty_scale * sigma_safe * sigma_safe
    
    # Combine EI and PI with variance penalty
    # Weight: 0.7 EI + 0.3 PI for balance between exploitation and stability
    base_score = 0.7 * ei + 0.3 * pi
    score = base_score - variance_penalty
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    score = score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
