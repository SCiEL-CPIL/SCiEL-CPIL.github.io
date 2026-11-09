import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Regret (ER) normalized by cost.
    # ER = E[max(f_best - f(x), 0)] which for a Gaussian posterior is:
    # ER = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This directly optimizes expected reduction in objective per unit cost.
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Regret
    # ER = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: when f_best - mu < 0, z < 0, Phi(z) is small, but the first term
    # becomes negative * small = small negative, and the second term is positive.
    # The overall ER should be non-negative since it's an expectation of max(0, f_best - f).
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    er = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure ER is non-negative (numerical safety)
    er = np.maximum(er, 0.0)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score
    score = er / cost_safe
    
    return score
