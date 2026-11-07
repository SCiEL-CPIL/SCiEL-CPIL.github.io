import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # Improvement = max(0, f_best - f(x))
    # E[Improvement] = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Avoid division by zero
    sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Where sigma is effectively 0, EI should be max(0, f_best - mu)
    # The formula above handles this correctly if we limit z, but 
    # let's explicitly set it for numerical stability in the limit.
    # If sigma is tiny, z is huge (positive if mu < f_best) or -huge (if mu > f_best).
    # If mu < f_best, z -> +inf, cdf -> 1, pdf -> 0. EI -> f_best - mu. Correct.
    # If mu > f_best, z -> -inf, cdf -> 0, pdf -> 0. EI -> 0. Correct.
    
    return ei
