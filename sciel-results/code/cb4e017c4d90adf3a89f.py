import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use Expected Improvement (EI) which is equivalent to Expected Hypervolume Improvement (HVI)
    # for a single objective minimization problem.
    # EI(x) = E[max(f_best - f(x), 0)]
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.where(sigma < 1e-8, 1e-8, sigma)
    
    # Calculate z-score for the improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF of the standard normal distribution
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Ensure cost is not zero
    cost_safe = np.clip(cost, 1e-8, None)
    
    scores = ei / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
