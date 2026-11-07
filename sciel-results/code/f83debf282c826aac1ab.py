import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses Expected Improvement (EI) for minimization.
    """
    # Deadline guard (though EI is fast, this is a safety measure)
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin

    # Input validation and sanitization
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and add small epsilon for numerical stability
    eps = 1e-8
    sigma = np.maximum(sigma, eps)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    # Since we are minimizing, we want mu < f_best.
    # Improvement is f_best - mu.
    z = (f_best - mu) / sigma
    
    # Standard EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF of standard normal, phi is PDF of standard normal.
    
    # Compute CDF and PDF
    # norm.cdf and norm.pdf are vectorized and fast
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI scores
    # Higher EI means more expected improvement (lower function value)
    scores = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure finite scores
    # If sigma is extremely small and mu < f_best, z is large positive, cdf->1, pdf->0, score->f_best-mu
    # If sigma is extremely small and mu > f_best, z is large negative, cdf->0, pdf->0, score->0
    # This behavior is correct.
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline (unlikely to be hit, but required by directive)
    if time.time() > deadline:
         # If we hit the deadline, we just return what we have.
         # The calculation above is O(N) and should be < 1ms for N=2000.
         pass

    return scores
