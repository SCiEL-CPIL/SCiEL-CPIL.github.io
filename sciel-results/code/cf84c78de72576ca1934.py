import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Time guard for safety, though this function is fast
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic Beta for UCB
    # We want to explore more when uncertainty is high relative to the model's predictive variation
    mu_scale = np.ptp(mu) + 1e-6
    sigma_mean = np.mean(sigma)
    
    # Beta starts at 2.0 and increases with relative uncertainty
    # Clip to prevent extreme exploration
    beta = 2.0 + 1.5 * (sigma_mean / mu_scale)
    beta = np.clip(beta, 1.5, 4.0)
    
    # Calculate UCB potential
    # We want to minimize mu. The lower confidence bound is mu - beta * sigma.
    # We want to pick points with low LCB.
    # Score should be high for low LCB.
    # Score = (f_best - LCB) / cost
    # LCB = mu - beta * sigma
    # Score = (f_best - mu + beta * sigma) / cost
    
    lcb = mu - beta * sigma
    potential = f_best - lcb
    scores = potential / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # If time is running out (unlikely for this simple op), return best effort
    # But since it's vectorized, it's fast.
    
    return scores
