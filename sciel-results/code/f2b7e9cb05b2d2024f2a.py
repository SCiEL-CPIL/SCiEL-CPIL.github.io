import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Expected Improvement
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Exploration bonus to escape local minima in low-sample regime
        beta = 1.0
        bonus = beta * sigma
        
        return ei + bonus
