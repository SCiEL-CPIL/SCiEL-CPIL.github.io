import numpy as np
from scipy.stats import norm
import time

_deadline = time.time() + 270.0

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition function for minimization
    # Use Expected Improvement (EI) style but adapted for cost
    # EI for minimization: E[max(f_best - f, 0)]
    # where f ~ N(mu, sigma)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # But we need to be careful: higher score = more likely to be evaluated
    # So we want high EI and low cost
    # Use EI / cost^alpha where alpha > 0 to penalize expensive points
    
    # Use a moderate cost penalty
    cost_penalty = cost ** 0.5
    
    scores = ei / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
