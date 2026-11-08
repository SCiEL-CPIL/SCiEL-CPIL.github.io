"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: total wall-clock limit is 330s, keep safety margin
_DEADLINE = 325.0  # 5s safety margin
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _log_norm_cdf(x):
    """Stable log of standard normal CDF."""
    # Use the identity: log(Phi(x)) for x < 0 can be computed via erfc
    # For x >= 0, log(Phi(x)) is close to 0 and can be computed directly
    x = np.asarray(x)
    result = np.empty_like(x, dtype=np.float64)
    mask_pos = x >= 0
    mask_neg = ~mask_pos
    
    # For x >= 0: Phi(x) is close to 1, log(Phi(x)) ≈ -0.5*x^2 + log(1 - ...)
    # More precisely, use scipy's special function or approximate
    # Actually, let's just use norm.logcdf which is stable
    result = norm.logcdf(x)
    return result


def _log_norm_pdf(x):
    """Log of standard normal PDF."""
    x = np.asarray(x)
    return -0.5 * x * x - 0.5 * np.log(2.0 * np.pi)


def _expected_improvement(mu, sigma, f_best):
    """
    Compute expected improvement in log-space for numerical stability.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    We compute log(EI) to avoid underflow.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # log(EI) = log((f_best - mu) * Phi(z) + sigma * phi(z))
    # This can be negative if f_best - mu < 0, but EI is always >= 0
    # Actually EI >= 0 always. Let's compute it directly but carefully.
    
    # Standard EI computation
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    return ei


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP standard deviations
    - f_best: scalar, best (lowest) observed value so far
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - score: (2000,) array of acquisition scores (higher = better to evaluate)
    """
    # Ensure start time is set
    _ensure_start()
    
    # Deadline check: if we're close to the deadline, return a simple score
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Return a simple exploration-based score
        score = sigma / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute expected improvement
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # But also add an exploration bonus for high uncertainty
    # The key insight: we want to balance exploitation (high EI) with
    # exploration (high sigma) and cost efficiency (low cost)
    
    # Use a weighted combination:
    # score = EI / cost + w * sigma / cost
    # This naturally handles cost awareness
    
    # Adaptive weight: when EI is small relative to sigma, explore more
    # When EI is large, exploit more
    ei_scale = np.maximum(ei, 1e-12)
    sigma_scale = sigma_safe
    
    # Compute a ratio to adapt the exploration weight
    # If EI >> sigma, we're in exploitation mode; if EI << sigma, exploration
    ratio = ei_scale / (sigma_scale + eps)
    
    # Smooth interpolation: weight for exploration decreases as ratio increases
    # Use a sigmoid-like function
    # w = 1 / (1 + ratio) gives w in (0, 1)
    # When ratio is small (EI << sigma), w ≈ 1 (explore)
    # When ratio is large (EI >> sigma), w ≈ 0 (exploit)
    w_explore = 1.0 / (1.0 + ratio)
    
    # Base weights
    w_exploit = 1.0
    w_explore_base = 0.5
    
    # Combined weight for exploration
    w_explore_final = w_exploit * w_explore_base * w_explore
    
    # Final score: (EI + w_explore * sigma) / cost
    # This is equivalent to a cost-aware UCB-like acquisition
    score = (ei + w_explore_final * sigma_safe) / np.maximum(cost, eps)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
