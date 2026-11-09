import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # Enhanced with a slight exploration boost to help escape local minima
    # in the low-budget regime (30 evaluations).
    
    # Handle zero or near-zero sigma to avoid numerical instability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Exploration factor: 1.0 is standard EI.
    # 1.1 slightly increases the weight of uncertainty (sigma) in the score,
    # encouraging exploration of less certain regions.
    exploration_factor = 1.1
    
    # Calculate the standardized improvement z using the scaled sigma
    # This effectively makes the GP appear more uncertain, boosting exploration.
    sigma_eff = exploration_factor * sigma_safe
    z = (f_best - mu) / sigma_eff
    
    # EI = (f_best - mu) * Phi(z) + sigma_eff * phi(z)
    # Note: Using sigma_eff in the second term and z calculation maintains consistency
    # with the "virtual" GP that has higher noise.
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_eff * phi
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
