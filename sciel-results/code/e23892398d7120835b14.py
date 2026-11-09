import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Epsilon to prevent issues with zero/negative sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Structural change: Use Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Compute the standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = CDF of standard normal
    # phi(z) = PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI for minimization
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI can be negative when f_best < mu (no improvement expected), but we take max with 0
    # Actually, the formula above already handles this correctly as Phi_z -> 0 when z -> -inf
    # But let's ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Use EI as the score (higher is better)
    scores = ei
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
         pass
         
    return scores
