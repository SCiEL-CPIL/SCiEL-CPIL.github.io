import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Dynamic Xi: Encourages exploration when f_best is large
    # As f_best decreases, xi decreases, favoring exploitation
    xi = 0.5 * fb
    
    # Effective best value for EI calculation
    fb_eff = fb - xi
    
    z = (fb_eff - mu) / sigma_safe
    
    # EI with xi
    ei = (fb_eff - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative values
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    if time.time() > _DEADLINE:
        return score
        
    return score
