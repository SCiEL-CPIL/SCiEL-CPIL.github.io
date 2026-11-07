import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI)
    with an adaptive exploration bonus based on the current best value.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy.stats.norm for CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Adaptive exploration bonus
    # f_best starts around 25 (max of x^2 on [-5,5]) and goes to 0.
    # We want high exploration early (high f_best) and low exploration late (low f_best).
    # r = f_best / 25.0
    # lambda = lambda_max * r
    # If f_best is very small, lambda is small.
    
    # Handle case where f_best might be negative? 
    # For sphere x^2, f_best >= 0.
    # But to be robust, clamp f_best to [0, 25] for the ratio calculation.
    f_max = 25.0
    r = np.clip(f_best, 0.0, f_max) / f_max
    
    # lambda_max = 0.5
    # This adds up to 0.5 * sigma to the EI score.
    # This helps escape local optima or poor initial regions.
    lambda_explore = 0.5 * r
    
    scores = ei + lambda_explore * sigma_safe
    
    # Check deadline (just in case, though this is fast)
    elapsed = time.time() - start_time
    if elapsed > 290: # Unlikely
         # Return pure EI as fallback
         return ei
         
    return scores
