import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track elapsed time and return early if needed
    # Since we don't have a global timer, we'll just ensure fast computation
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness: Divide by cost to penalize expensive evaluations
    # Cost is in [1, 2]. Use a power > 1 to more strongly penalize expensive points
    # Also add a small exploration term based on sigma
    cost_safe = np.maximum(cost, 1e-9)
    
    # Primary term: EI per unit cost
    score = ei / cost_safe
    
    # Exploration bonus: encourage exploring uncertain regions, weighted by cost efficiency
    # This helps escape local optima in the multimodal Levy function
    exploration = 0.05 * sigma_safe / cost_safe
    
    score = score + exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
