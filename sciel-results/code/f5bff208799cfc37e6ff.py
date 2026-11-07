import numpy as np
from scipy.special import ndtr

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    Uses scipy.special.ndtr for the CDF and manual calculation for the PDF
    to improve numerical stability and performance.
    """
    # Avoid division by zero and ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # m = f_best - mu
    # s = sigma
    m = f_best - mu
    s = sigma_safe
    
    # z = (f_best - mu) / sigma
    z = m / s
    
    # CDF of standard normal: Phi(z)
    # ndtr(z) is the CDF of the standard normal distribution
    cdf = ndtr(z)
    
    # PDF of standard normal: phi(z)
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # EI = m * Phi(z) + s * phi(z)
    ei = m * cdf + s * pdf
    
    # Ensure EI is non-negative (numerical errors might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    return ei
