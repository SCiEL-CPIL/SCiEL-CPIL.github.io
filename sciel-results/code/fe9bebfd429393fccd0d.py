import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    
    # Handle sigma == 0 to avoid division by zero
    # If sigma is 0, EI is max(f_best - mu, 0)
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in norm.cdf/pdf
    # Though norm.cdf/pdf are generally stable, clipping is a good practice.
    # z > 8 is essentially 1.0 for cdf, 0.0 for pdf.
    # z < -8 is essentially 0.0 for cdf, 0.0 for pdf.
    z_clipped = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure non-negative EI (numerical errors might cause tiny negatives)
    ei = np.maximum(ei, 0)
    
    return ei
