"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 5.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time) - _MARGIN

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    divided by cost.
    
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    This formulation captures the magnitude of expected improvement.
    Dividing by cost makes it cost-aware.
    """
    # Deadline guard
    if _remaining_time() < 1.0:
        # Fast fallback: simple PI / cost
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        cost_safe = np.maximum(cost, 1e-8)
        scores = pi / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive and handle numerical stability
    # If sigma is very small, EI is approx max(f_best - mu, 0)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z
    z = (f_best - mu) / sigma_safe
    
    # Compute EI components
    # norm.cdf and norm.pdf are vectorized
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI formula
    # (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # For very small sigma, the above formula can be numerically unstable or inaccurate
    # In the limit sigma -> 0, EI -> max(f_best - mu, 0)
    # We can use a mask to replace EI where sigma is effectively zero
    # However, with sigma_safe >= 1e-8, the formula is usually stable enough.
    # But let's be safe: if sigma is tiny, the improvement is deterministic if mu < f_best.
    # Actually, the standard EI formula is robust for small sigma if computed carefully.
    # scipy.stats.norm.cdf and pdf handle this reasonably well.
    
    # Cost safety
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware score: EI / cost
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
