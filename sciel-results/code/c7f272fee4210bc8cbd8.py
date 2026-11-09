"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: kill after 330s wall-clock, return with safety margin
_DEADLINE = time.time() + 325.0  # 5s safety margin before 330s limit


def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    # Use stable computation
    ei = np.zeros_like(mu)
    mask = z > 0
    ei[mask] = (f_best[mask] - mu[mask]) * _norm_cdf(z[mask]) + sigma[mask] * _norm_pdf(z[mask])
    # For z <= 0, EI is 0 (or very small negative, clip to 0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    scores = ei / np.maximum(cost, 1e-12)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores


def _norm_cdf(x):
    """Standard normal CDF using scipy or manual."""
    try:
        from scipy.stats import norm
        return norm.cdf(x)
    except ImportError:
        # Fallback using error function
        return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
# EVOLVE-BLOCK-END
