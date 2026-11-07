import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = 290.0  # Safety margin from 300s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Lower Confidence Bound (LCB) score for minimization
    # We want to maximize this score to find the next point.
    # LCB = mu - beta * sigma. We want the lowest LCB.
    # So Score_LCB = -LCB = -mu + beta * sigma.
    beta = 2.0  # Exploration parameter
    lcb_score = -mu + beta * sigma_safe
    
    # Combine EI and LCB
    # Using max ensures we either exploit a known improvement (EI) 
    # or explore a region with high potential (LCB).
    scores = np.maximum(ei, lcb_score)
    
    # Check time
    if time.time() - start_time > deadline:
        # Return best effort, usually EI is safer if time is tight
        return ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
