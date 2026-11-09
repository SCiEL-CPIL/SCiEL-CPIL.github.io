"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track time and potentially seed if needed
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize time tracking if not already set
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Safety margin: return valid scores immediately if close to limit
    # The evaluator kills after 330s, task limit is 300s.
    # If we are past 290s, just return something valid (e.g., 0 or mu) to finish quickly.
    if elapsed > 290.0:
        # Return a neutral score to avoid crashing, effectively ending search or making a random pick
        # Since we need to return 2000 scores, we return zeros or mu.
        # Returning mu is safer than 0 if mu has scale, but 0 is fine as long as it's finite.
        return np.zeros_like(mu)

    # Strategy: Use Expected Improvement (EI) for minimization.
    # This is a standard and robust acquisition function for Bayesian Optimization.
    # The current program uses a heuristic LCB with fixed kappa based on f_best thresholds.
    # EI adapts automatically to the uncertainty (sigma) and improvement potential (f_best - mu).
    #
    # For minimization, the improvement is defined as I(x) = max(0, f_best - f(x)).
    # The Expected Improvement is E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z),
    # where z = (f_best - mu) / sigma, Phi is the standard normal CDF, and phi is the standard normal PDF.
    
    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Note: We use the standard normal CDF and PDF from scipy.stats
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return ei
# EVOLVE-BLOCK-END
