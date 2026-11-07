import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
       eps = 1e-12
       sigma_safe = np.maximum(sigma, eps)
       
       # Expected Improvement for minimization
       # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
       
       z = (f_best - mu) / sigma_safe
       
       # Avoid overflow/underflow in pdf/cdf for extreme z
       # norm.cdf and norm.pdf are stable enough for typical z
       
       ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
       
       # Ensure finite
       ei = np.where(np.isfinite(ei), ei, 0.0)
       
       return ei
