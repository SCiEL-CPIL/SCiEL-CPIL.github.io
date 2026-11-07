import numpy as np
import time
from scipy.stats import norm

# Global time tracking
_start_time = time.time()
TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI/C) acquisition function.
    
    Maximizes the expected improvement per unit of cost.
    EI(x) = E[max(f_best - f(x), 0)]
    Score = EI(x) / cost(x)
    
    This is a standard, robust approach for cost-aware Bayesian optimization.
    """
    # Check time
    current_time = time.time()
    if current_time - _start_time > TIME_LIMIT - 5:
        # Emergency fallback: favor low mu and low cost
        cost_safe = np.maximum(cost, 1e-8)
        return -mu / cost_safe

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)

    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best < mu, z is negative, Phi(z) is small, EI is small (but positive due to sigma*phi(z))
    # If f_best > mu, z is positive, Phi(z) is large, EI is large.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score
    score = EI / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
