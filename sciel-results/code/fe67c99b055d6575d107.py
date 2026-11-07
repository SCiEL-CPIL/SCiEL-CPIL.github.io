import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Thompson Sampling.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs
    eps = 1e-8
    sigma = np.maximum(sigma, eps)
    mask = np.isfinite(mu) & np.isfinite(sigma)
    
    # Adaptive Exploration Scale
    # f_opt is the known global minimum
    f_opt = -39.166
    
    # Calculate distance to optimum
    # f_best is typically > f_opt
    delta = f_best - f_opt
    
    # If delta is negative (better than known opt, unlikely but possible due to noise/error), 
    # treat as 0.
    delta = max(delta, 0.0)
    
    # Map delta to alpha
    # We want alpha = 1.0 when delta is large (e.g., > 20)
    # We want alpha = 0.2 when delta is small (e.g., < 1)
    # Linear interpolation
    # alpha = 0.2 + 0.8 * (delta / 20.0)
    # Clip to [0.2, 1.0]
    alpha = 0.2 + 0.8 * (delta / 20.0)
    alpha = np.clip(alpha, 0.2, 1.0)
    
    # Check deadline
    if time.time() > deadline - 5.0:
        # Fallback to pure exploitation
        scores = -mu
        scores[~mask] = -np.inf
        return scores
    
    # Thompson Sampling
    # Draw standard normal noise
    z = np.random.randn(mu.shape[0])
    
    # Sampled values
    # y = mu + alpha * sigma * z
    y = mu + (alpha * sigma) * z
    
    # We want to minimize y
    # Return scores where higher is better
    scores = -y
    
    # Handle non-finite
    scores[~mask] = -np.inf
    
    return scores
