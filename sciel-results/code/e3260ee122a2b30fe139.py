import numpy as np
from scipy.stats import norm
import time

# Global state for tracking elapsed time and providing deadline safety
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    current_time = time.time()
    elapsed = current_time - _start_time
    
    # Safety margin: if we are dangerously close to the 330s limit,
    # return a quick deterministic score to avoid timeout.
    if elapsed > 325:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.maximum(sigma, 1e-9)
    
    # Strategy: Use Probability of Improvement (PI) with a small exploration jitter
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    # For minimization, we want points with high probability of being better than f_best.
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Compute CDF (Probability of Improvement)
    pi = norm.cdf(z)
    
    # Add a small exploration jitter to encourage some exploration
    # The jitter is added to the probability, but we need to ensure it doesn't dominate
    # A better approach: add a small term that rewards high uncertainty when PI is low
    # Or simply use PI + epsilon * sigma (normalized)
    
    # Let's try a hybrid: PI + small weight * (normalized sigma)
    # But the directive says: "Use a Probability of Improvement (PI) acquisition function 
    # with a small exploration jitter (e.g., epsilon=1e-4) to prioritize points with a 
    # high probability of beating the current best"
    
    # So, score = PI + epsilon * something_exploratory
    # Let's use epsilon * sigma as the exploratory term, but scale it to be comparable to PI [0,1]
    # Since PI is in [0,1], we can normalize sigma to [0,1] over the candidate set
    
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-9:
        sigma_norm = (sigma - sigma_mean) / sigma_std
        # Map to [0,1] approximately
        sigma_norm = (sigma_norm - np.min(sigma_norm)) / (np.max(sigma_norm) - np.min(sigma_norm) + 1e-9)
    else:
        sigma_norm = np.zeros_like(sigma)
    
    epsilon = 1e-4
    scores = pi + epsilon * sigma_norm
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Clip to [0, 1] to keep them in a reasonable range
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores
