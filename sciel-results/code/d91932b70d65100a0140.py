import numpy as np
import time
from scipy.stats import norm

_T0 = None
_DEADLINE = None
_MARGIN = 5.0
_call_count = 0

def _ensure_deadline():
    global _DEADLINE, _T0
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 300.0
    return _DEADLINE

def acquisition(mu, sigma, f_best):
    global _call_count
    deadline = _ensure_deadline()
    n = mu.shape[0]
    
    # Increment call count
    _call_count += 1
    
    # Time guard
    elapsed = time.perf_counter() - _T0
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: pure exploitation (minimize mu)
        return -mu

    # Clamp sigma to avoid division by zero or numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # Improvement is f_best - f(x)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, if mu < f_best, z > 0, Phi(z) is close to 1, EI is large.
    # If mu > f_best, z < 0, Phi(z) is small, EI is small (unless sigma is large).
    
    # Use scipy.stats.norm for Phi and phi
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
