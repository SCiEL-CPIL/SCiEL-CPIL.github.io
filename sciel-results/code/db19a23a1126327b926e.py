import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    # The total budget is 330s. We want to be safe.
    if time.time() - _START_TIME > 325:
        # Fallback: simple EI / cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    epsilon = 1e-9
    
    # Calculate Z-score
    z = (f_best - mu) / (sigma + epsilon)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Scale PI to match EI units
    # Using median sigma as a robust scale for the current uncertainty
    med_sigma = np.median(sigma)
    
    # Blend
    # We use a weighted sum. 
    # The previous success used "normalizing both". 
    # Let's try standardizing EI and PI across the 2000 candidates to ensure 
    # they contribute equally regardless of the current GP state.
    # This is a stronger structural change than just scaling by median sigma.
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_std = 1e-9
        
    pi_mean = np.mean(pi)
    pi_std = np.std(pi)
    if pi_std < 1e-9:
        pi_std = 1e-9
        
    ei_norm = (ei - ei_mean) / ei_std
    pi_norm = (pi - pi_mean) / pi_std
    
    # Combine
    # We want higher scores to be better.
    # Standardized values can be negative. 
    # Cost division amplifies negative scores (making them more negative).
    # This is fine, as we want to avoid high cost points that don't have high acquisition.
    
    combined = ei_norm + pi_norm
    
    scores = combined / (cost + epsilon)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
