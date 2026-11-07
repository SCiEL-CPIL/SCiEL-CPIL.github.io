import numpy as np
import time

# Track time to ensure we don't exceed the 330s limit (with safety margin)
_start_time = None
_DEADLINE = 320.0  # seconds, safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) relative to f_best for exploitation
    2. A constant exploration bonus based on known minimum (3.0) to maintain
       sensitivity to high-uncertainty regions
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher meaning more promising.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return early if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return simple UCB-like scores
        return -mu + sigma
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    # Use scipy's norm or compute manually
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-z^2/2) / sqrt(2*pi)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    phi_z = np.exp(-0.5 * z**2) / sqrt_2pi
    # Numerically stable CDF
    from scipy.stats import norm
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Add a constant exploration bonus to maintain sensitivity to uncertain regions
    # This prevents the acquisition from becoming too exploitative as f_best approaches 3.0
    exploration_weight = 0.5
    exploration_bonus = exploration_weight * sigma
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
