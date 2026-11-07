import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start = time.time()
    limit = 1.0 # 1 second limit per call
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Quantile Acquisition Function (QAF)
    # Maximizes the probability of finding a point with f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Check time
    if time.time() - start > limit:
         # Fallback to a simpler metric if cdf was slow (unlikely)
         # But cdf is vectorized and fast.
         pass
         
    return scores
