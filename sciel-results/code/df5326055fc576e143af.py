import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Cost-aware UCB for minimization
        # We want to minimize f(x).
        # LCB = mu - beta * sigma
        # We want to select x with low LCB.
        # Score = -LCB / cost = (beta * sigma - mu) / cost
        
        beta = 3.0
        
        # Ensure sigma is positive to avoid issues, though GP sigma should be > 0
        sigma_safe = np.maximum(sigma, 1e-9)
        
        lcb = mu - beta * sigma_safe
        score = -lcb / cost
        
        # Mask invalid values
        score = np.where(np.isfinite(score), score, -np.inf)
        
        return score
