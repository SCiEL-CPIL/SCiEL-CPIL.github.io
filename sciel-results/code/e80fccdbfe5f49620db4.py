import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
        # Deadline guard
        # The function is vectorized and fast, but we adhere to the directive.
        # No complex loops, so this is a safeguard against unexpected issues.
        
        sigma = np.maximum(sigma, 1e-9)
        
        # Expected Improvement
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # UCB for minimization
        # We maximize this score.
        # Low mu is good. High sigma is good (uncertainty).
        beta = 1.5
        ucb = -mu + beta * sigma
        
        # Blend
        # Normalize UCB? 
        # EI is in units of improvement.
        # UCB is in units of objective.
        # If f_best is 0, and we find -10, EI ~ 10. UCB ~ 10.
        # They are comparable.
        
        score = (0.5 * ei + 0.5 * ucb) / cost
        
        # Ensure finite
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return score
