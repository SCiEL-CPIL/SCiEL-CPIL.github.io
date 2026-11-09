import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Structural Change: Pareto-frontier guided exploration
    # 1. Identify anchor: candidate with lowest predicted mean
    idx_best = np.argmin(mu)
    anchor_mu = mu[idx_best]
    anchor_sigma = sigma[idx_best]
    
    # 2. Compute Euclidean distance in prediction space (mu, sigma)
    # Note: mu and sigma are in objective units.
    # We scale sigma to be comparable to mu if necessary, but let's try raw first.
    # If scales differ vastly, distance is dominated by mu.
    # Let's normalize sigma by its max value to balance the distance metric.
    max_sigma = np.max(sigma)
    if max_sigma < 1e-8:
        max_sigma = 1e-8
        
    mu_scaled = mu
    sigma_scaled = sigma / max_sigma
    
    dist = np.sqrt((mu_scaled - anchor_mu)**2 + (sigma_scaled - anchor_sigma/max_sigma)**2)
    
    # 3. Normalize distance
    max_dist = np.max(dist)
    if max_dist < 1e-8:
        max_dist = 1e-8
    norm_dist = dist / max_dist
    
    # 4. Score: product of sigma and normalized distance
    # This promotes high uncertainty regions that are distinct from the current best prediction
    scores = sigma * norm_dist
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
