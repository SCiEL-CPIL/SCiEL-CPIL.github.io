import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Ensure sigma is positive for division
        sigma_safe = np.maximum(sigma, 1e-9)
        
        # Standard Expected Improvement
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        
        # Exploration bonus: encourage high uncertainty
        # Using a small coefficient to balance exploitation (EI) and exploration (sigma)
        beta = 0.5
        exploration_bonus = beta * sigma_safe
        
        # Combined score before cost adjustment
        raw_score = ei + exploration_bonus
        
        # Cost adjustment with power law
        alpha = 0.7
        # cost is provided, ensure it's positive
        cost_safe = np.maximum(cost, 1e-9)
        
        scores = raw_score / (cost_safe ** alpha)
        
        # Ensure finite values
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
