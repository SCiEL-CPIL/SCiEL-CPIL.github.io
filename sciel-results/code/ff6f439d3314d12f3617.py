"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for potential timeout guards if needed, 
# though simple vectorized ops are very fast.
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
        We return the Expected Improvement for minimization.
    """
    # Calculate the improvement over the best observed value
    # For minimization, improvement is f_best - mu
    improvement = f_best - mu
    
    # Normalize by standard deviation
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    z = improvement / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores (though EI should be finite for finite inputs)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return ei
# EVOLVE-BLOCK-END
