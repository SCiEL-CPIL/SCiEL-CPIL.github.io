import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill

def _time_remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _sanitize(scores):
    """Ensure scores are valid: finite, correct shape."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI-LCB acquisition with deadline guard.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed function value so far.
        
    Returns:
        scores: Array of shape (2000,) where higher scores are preferred.
    """
    # Deadline guard
    if _time_remaining() < 5.0:
        # Return simple EI as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return _sanitize(ei)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = float(np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6))
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best_safe - mu_safe) / sigma_safe
    ei = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate LCB (Lower Confidence Bound) for minimization
    # We want to minimize mu - kappa * sigma
    # Score should be high when mu - kappa * sigma is low
    # So score = -(mu - kappa * sigma) = -mu + kappa * sigma
    # Use a moderate kappa to balance exploration and exploitation
    kappa = 1.5
    lcb = -mu_safe + kappa * sigma_safe
    
    # Hybrid strategy: blend EI and LCB
    # Weight towards EI for exploitation, LCB for exploration
    w_ei = 0.6
    w_lcb = 0.4
    
    final_scores = w_ei * ei + w_lcb * lcb
    
    # Check time
    if _time_remaining() < 10.0:
        return _sanitize(final_scores)
    
    # Ensure finite
    final_scores = _sanitize(final_scores)
    
    return final_scores
