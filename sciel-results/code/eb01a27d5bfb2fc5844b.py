import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Cost-aware Expected Improvement (EI) with softer cost penalty
        # Score = EI(x) / sqrt(cost(x))
        
        # Ensure cost is positive
        cost_safe = np.maximum(cost, 1e-8)
        
        # Calculate z-score
        z = (f_best - mu) / (sigma + 1e-9)
        
        # Expected Improvement for minimization
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Normalize by square root of cost
        # This is less aggressive than 1/cost, potentially allowing
        # exploration of higher-cost regions if the GP predicts high EI there.
        scores = ei / np.sqrt(cost_safe)
        
        # Ensure finite scores
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        return scores
