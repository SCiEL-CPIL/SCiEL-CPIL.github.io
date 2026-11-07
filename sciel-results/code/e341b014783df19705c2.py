import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        # Clip sigma
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        
        # Dynamic Beta based on f_best?
        # If f_best is large, we are far from optimum, explore more.
        # If f_best is small, exploit.
        # f_best starts high (e.g. 100) and goes to 0.
        # Let's use a simple scaling.
        # beta = 2.0 + 2.0 * (1 / (1 + f_best/10)) ?
        # Or just fixed.
        
        beta = 2.0
        
        # UCB for minimization: mu - beta * sigma
        # We want to maximize the potential for low values.
        # Score = (beta * sigma - mu) / cost
        scores = (beta * sigma_safe - mu) / cost_safe
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        return scores
