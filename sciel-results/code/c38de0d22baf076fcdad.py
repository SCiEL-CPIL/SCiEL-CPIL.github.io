"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Aware Expected Improvement (cEI).
    
    1. Compute Expected Improvement (EI) for minimization:
       EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
       where Z = (f_best - mu) / sigma, Phi is CDF and phi is PDF of standard normal.
    2. Compute Cost-Aware EI:
       cEI = EI / cost
    3. Return cEI scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero in Z
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # f_best is the lowest observed value (minimization)
    # If mu < f_best, Z is positive, indicating potential improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * P(Y < f_best) + E[f_best - Y | Y < f_best] * P(Y < f_best) ... simplified formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    phi_z_cdf = norm.cdf(z)
    
    ei = (f_best - mu) * phi_z_cdf + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
