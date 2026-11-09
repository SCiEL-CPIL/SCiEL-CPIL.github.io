import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    
    # Handle zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI components
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI) - more exploratory than EI
    # PoI = Phi(z) = probability that f(x) < f_best
    poi = cdf_z
    
    # Blend EI and PoI: PoI is more exploratory, reducing risk of local optima
    # in the multimodal Shekel 4d function within the tight budget
    # Weight: 0.5 * EI + 0.5 * PoI * scale_factor
    # Scale PoI to be comparable to EI magnitude
    # EI can be large when f_best - mu is large, PoI is in [0, 1]
    # Use a scaling that makes them comparable
    ei_scale = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    poi_scaled = poi * ei_scale
    
    # Blend: 60% EI, 40% PoI for a balance of exploitation and exploration
    blended = 0.6 * ei + 0.4 * poi_scaled
    
    # Cost-aware score
    # Cost is in [1, 2]. Use quadratic penalty for stronger preference for low-cost points
    cost_safe = np.maximum(cost, 1e-9)
    scores = blended / (cost_safe ** 2)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
