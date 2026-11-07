import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        # Fallback: Fixed LCB (strong exploration)
        return -mu - 2.0 * np.maximum(sigma, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (1-step Knowledge Gradient) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # f(x) ~ N(mu, sigma^2)
    # Let Z = (f(x) - f_best) / sigma
    # EI = (f_best - mu) * P(f(x) < f_best) + sigma * pdf(Z)
    # Note: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI calculation
    # If z is very large (mu << f_best), cdf->1, pdf->0, EI-> f_best - mu
    # If z is very small (mu >> f_best), cdf->0, pdf->0, EI-> 0
    scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
