import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the 330s limit, return a safe exploitation-based score
    # to ensure we don't timeout. 300s is a safe margin.
    if time.time() - _START_TIME > 300:
        # Fallback: exploit best known, penalized by cost
        # Use -mu/cost to prioritize low values and low costs
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback = -mu / np.maximum(cost, 1e-9)
        # Replace non-finite with a very low score
        fallback = np.where(np.isfinite(fallback), fallback, -1e9)
        return fallback

    # Cost-Adjusted Lower Confidence Bound (LCB)
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so maximize -LCB(x) / cost(x)
    # Score(x) = (beta * sigma(x) - mu(x)) / cost(x)
    
    # Beta controls exploration. 
    # A value of 1.5 is a reasonable balance for 4D multimodal functions.
    beta = 1.5
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 1e9) # If mu is inf, score will be -inf (bad)
    
    # Calculate scores
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = (beta * sigma - mu) / np.maximum(cost, 1e-9)
        
    # Handle any remaining non-finite values
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
