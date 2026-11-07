import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_MODULE_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    Modified with an exploration parameter xi to encourage larger improvements.
    """
    # Deadline guard: if we are close to the 330s limit, return a safe, fast score
    if time.time() - _MODULE_START_TIME > 300:
        # Fallback: Return sigma (pure exploration) to ensure we don't crash
        return np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    # Expected Improvement for minimization with target shift xi
    # We want to maximize E[ max(f_best + xi - f(x), 0) ]
    # This encourages finding points significantly better than the current best.
    
    # Handle sigma = 0 cases to avoid division by zero
    eps = 1e-10
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Exploration parameter xi
    # A small positive xi makes the acquisition function more exploratory
    # by requiring a larger improvement to get a high score.
    xi = 0.1
    
    # Target value
    target = f_best + xi
    
    # Z-score: (target - mu) / sigma
    z = (target - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula with shifted target
    ei = (target - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite outputs
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
