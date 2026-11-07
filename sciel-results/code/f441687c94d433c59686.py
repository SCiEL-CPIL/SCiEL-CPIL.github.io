import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 1.0  # 1 second is more than enough
    
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)
    sigma = np.maximum(sigma, 1e-12)
    
    # Check deadline early
    if time.time() > deadline:
        return np.zeros_like(mu)
        
    # Calculate tolerance based on median uncertainty
    # This allows exploring points slightly worse than f_best if they have high uncertainty
    tol = np.median(sigma)
    
    # Pareto-frontier filtering:
    # Keep candidates where mu is not too much worse than f_best
    mask = mu < (f_best + tol)
    
    # Pure UCB score (maximized for minimization problem)
    # score = -mu + beta * sigma
    beta = 2.0  # Aggressive exploration factor
    
    raw_score = -mu + beta * sigma
    
    # Apply mask: invalid points get very low score
    scores = np.where(mask, raw_score, -1e9)
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
