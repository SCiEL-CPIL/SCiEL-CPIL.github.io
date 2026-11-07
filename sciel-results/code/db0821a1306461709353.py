import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (EI) acquisition function.
    
    This strategy uses Expected Improvement, which favors points with high potential
    for large improvements, not just any improvement (unlike PoI). 
    To account for the varying evaluation costs, the EI score is normalized by the 
    cost of evaluating the point. This encourages the optimizer to select points 
    that offer the best "bang for the buck" in terms of expected function value 
    reduction per unit cost.
    
    Parameters:
    mu: array of shape (2000,) - Mean of the GP posterior
    sigma: array of shape (2000,) - Standard deviation of the GP posterior
    f_best: float - Best observed value so far
    cost: array of shape (2000,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - Acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is norm.cdf, phi is norm.pdf
    phi_z = norm.pdf(z_clipped)
    cdf_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization
    # Avoid division by zero (cost is at least 1.0 + 0 = 1.0, so safe, but good practice)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score is EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
