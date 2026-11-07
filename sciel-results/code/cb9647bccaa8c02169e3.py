import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Ensure inputs are arrays
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        cost = np.asarray(cost)
        
        # Avoid division by zero
        sigma_safe = np.maximum(sigma, 1e-9)
        
        # z-score for EI
        z = (f_best - mu) / sigma_safe
        
        # Expected Improvement
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        
        # Exploration bonus
        # The prompt indicates sigma * 0.05 was effective. 
        # Let's use 0.1 to encourage more exploration in 4D.
        exploration = 0.1 * sigma_safe
        
        # Total value before cost adjustment
        total_value = ei + exploration
        
        # Cost adjustment
        # Ensure cost is positive and finite
        cost_safe = np.maximum(cost, 1e-9)
        
        scores = total_value / cost_safe
        
        # Handle non-finite values
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
