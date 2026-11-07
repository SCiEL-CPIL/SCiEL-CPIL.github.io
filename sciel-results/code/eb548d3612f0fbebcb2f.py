"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    EI balances exploration and exploitation by calculating the expected 
    improvement over the current best f_best. Higher EI values indicate 
    candidates more likely to improve upon the current best.
    
    Formula:
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    """
    # Deadline guard (though EI is very fast for 2000 points)
    start_time = time.time()
    
    # Convert inputs to numpy arrays for efficiency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and safe for division
    # Use a small epsilon to avoid division by zero if sigma is 0
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    # z represents how many standard deviations f_best is above the mean
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    # (f_best - mu) is the potential improvement if we were to sample at x
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical artifacts: if mu > f_best and sigma is tiny, EI should be ~0
    # The formula naturally handles this, but we can clean up negative values 
    # (EI should theoretically be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Ensure scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 295:
        # In the extremely unlikely event of timeout, return current best
        pass
        
    return ei
# EVOLVE-BLOCK-END
