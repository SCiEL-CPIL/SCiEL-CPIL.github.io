import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses the 1-step Knowledge Gradient (KG) approximation, which is equivalent
    to Expected Improvement (EI) for minimization. This is a structural change
    from the previous hybrid normalized EI+UCB approach, relying directly on the
    probabilistic properties of the Gaussian Process posterior.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate z-score for EI
    # f_best is the current minimum (lower is better)
    # Improvement is f_best - Y
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical underflow/overflow in cdf/pdf
    # norm.cdf(-10) ~ 7e-24, norm.pdf(-10) ~ 7e-24
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement (1-step Knowledge Gradient)
    # EI = E[max(f_best - Y, 0)]
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Time guard
    if time.time() - start_time > 290:
        # Fallback to a simple UCB-like score if computation is unexpectedly slow
        return -mu + 0.1 * sigma

    return ei
