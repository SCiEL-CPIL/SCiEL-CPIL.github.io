"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None
_bo_call_count = 0

def acquisition(mu, sigma, f_best):
    global _bo_start_time, _bo_call_count
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    _bo_call_count += 1
    
    # Deadline Guard: 330s limit, use 305s as safety margin
    # If we are close to timeout, prioritize speed and exploitation
    # to ensure the loop finishes.
    if time.time() - _bo_start_time > 305:
        # Return exploitation score: lower mu is better, so -mu is higher score
        return -np.asarray(mu, dtype=np.float64)
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Structural Change: Expected Improvement (EI)
    # EI is a deterministic acquisition function that balances exploration and exploitation.
    # It calculates the expected improvement over the current best value f_best.
    # For minimization, improvement is max(0, f_best - f(x)).
    
    # Clip sigma to avoid division by zero and numerical instability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # If sigma is very small, Z will be large if mu < f_best, small if mu > f_best
    Z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    
    # Vectorized calculation
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    ei_scores = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Handle the case where sigma is effectively zero (deterministic prediction)
    # In this case, EI is simply max(0, f_best - mu)
    # The formula above with sigma_safe=eps approximates this, but let's be explicit for clarity/safety
    # Actually, if sigma is 0, Z is inf or -inf. 
    # If mu < f_best, Z = inf, Phi=1, phi=0 -> EI = f_best - mu. Correct.
    # If mu > f_best, Z = -inf, Phi=0, phi=0 -> EI = 0. Correct.
    # If mu == f_best, Z = 0, but sigma=0 is problematic. 
    # Using sigma_safe=eps handles this gracefully as Z becomes large magnitude.
    
    # Ensure finite values
    scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
