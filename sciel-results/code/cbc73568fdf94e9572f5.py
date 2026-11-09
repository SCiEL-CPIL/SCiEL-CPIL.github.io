import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # This models the expected magnitude of improvement, providing a better
    # multi-objective trade-off signal than POI which only models probability.
    phi_z = norm.pdf(z)
    # Clip phi_z to avoid numerical underflow issues in extreme tails
    phi_z = np.clip(phi_z, 1e-12, None)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * phi_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # This approximates the Pareto-optimal trade-off between expected
    # regret reduction and evaluation cost.
    scores = ei / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
