"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return best observed (greedy) or zeros
        # Greedy is safe
        return -mu

    # Expected Hypervolume Ratio (EHypR) for minimization
    # EHypR = E[max(f_best, f(x)) / f(x)]
    # For f(x) > 0: EHypR = f_best/f(x) * P(f(x) < f_best) + P(f(x) >= f_best)
    #              = f_best * phi(z) / sigma * Phi(z) + (1 - Phi(z))
    # where z = (f_best - mu) / sigma
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EHypR = f_best * phi(z) / sigma * Phi(z) + (1 - Phi(z))
    # Note: For f(x) <= 0, we need to handle carefully, but for sphere function on [-5,5]
    # with minimum 0, f(x) >= 0 always
    
    # However, if f_best is close to 0, we need to be careful
    # EHypR simplifies when f_best is small
    ehyp = f_best * phi * Phi / sigma_safe + (1 - Phi)
    
    # Ensure finite
    if not np.all(np.isfinite(ehyp)):
        ehyp = np.nan_to_num(ehyp, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return ehyp
# EVOLVE-BLOCK-END
