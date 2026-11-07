import numpy as np
import time

# Global state to track evaluation count for adaptive beta
_eval_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
        
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising points.
    """
    global _eval_count, _start_time
    
    # Deadline guard
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Return a safe fallback if we're too close to the limit
        return np.zeros_like(mu)
    
    _eval_count += 1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF of the standard normal and phi is the PDF
    
    from scipy.stats import norm
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative EI (numerical safety)
    ei = np.maximum(ei, 0)
    
    # Add a small exploration term to break ties and encourage diversity
    # This helps escape local optima in the multimodal Hartmann 3D landscape
    # The exploration term is scaled by sigma to be larger in uncertain regions
    exploration = 0.1 * sigma
    
    # Combine EI with exploration
    scores = ei + exploration
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
