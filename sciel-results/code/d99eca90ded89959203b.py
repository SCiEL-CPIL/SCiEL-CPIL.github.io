import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Uses standard EI (Expected Improvement) which is deterministic and
    well-understood, with cost-awareness applied via division by cost.
    This is a structural change from Thompson Sampling to a deterministic
    exploitation-focused approach with explicit uncertainty quantification.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviations.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the 330s limit, return a safe
    # fallback (pure exploitation via mu) to avoid timeout.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        mu_arr = np.asarray(mu)
        fallback_scores = -mu_arr
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety clamps
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Standard Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle the case where sigma is very small (deterministic region)
    # In this case, EI is simply max(f_best - mu, 0)
    eps = 1e-8
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Compute EI components
    phi_z = norm.pdf(z_clipped)  # Standard normal PDF
    Phi_z = norm.cdf(z_clipped)  # Standard normal CDF
    
    # EI formula for minimization
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness: divide by cost to penalize expensive evaluations
    # This is a simple but effective cost-awareness mechanism
    scores = ei / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
