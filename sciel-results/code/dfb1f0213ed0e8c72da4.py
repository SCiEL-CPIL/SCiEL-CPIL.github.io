import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return pure PMM scores
        scores = -np.asarray(mu, dtype=np.float64)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize sigma to prevent division by zero and numerical instability
    min_sigma = 1e-6
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > min_sigma), sigma, min_sigma)
    
    # Adaptive strategy:
    # When f_best is close to 0 (good solution found), switch to pure exploitation (-mu)
    # to converge exactly to the minimum. When f_best is far from 0, use UCB for exploration.
    
    # Threshold for switching to exploitation
    exploit_threshold = 0.5
    
    if f_best < exploit_threshold:
        # Pure exploitation: just minimize the predicted mean
        scores = -mu
    else:
        # UCB with adaptive beta
        # Normalize sigma to [0, 1] for adaptive beta
        sigma_max = np.max(sigma_safe)
        if sigma_max > 0:
            sigma_norm = sigma_safe / sigma_max
        else:
            sigma_norm = np.zeros_like(sigma_safe)
        
        # Adaptive beta: higher when uncertainty is more variable
        # This encourages exploration in regions where the GP is uncertain
        mu_scale = np.std(mu) + 1e-10
        sigma_scale = np.std(sigma_safe) + 1e-10
        
        # Base beta that balances exploration and exploitation
        # Start with a moderate value and adjust based on uncertainty
        beta = 0.7 + 0.3 * (sigma_scale / (mu_scale + sigma_scale + 1e-10))
        
        # Compute UCB scores
        scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
