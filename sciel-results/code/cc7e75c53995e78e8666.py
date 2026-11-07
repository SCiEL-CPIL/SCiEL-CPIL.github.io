import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    Uses Expected Improvement (EI) with multiplicative cost discount.
    """
    # Track time to avoid timeout
    t_start = time.time()
    deadline = t_start + 290.0  # safety margin before 300s limit

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma to be non-negative and finite
    sigma = np.clip(sigma, 0.0, None)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z) where Z = (f_best - mu) / sigma
    # When sigma is very small, EI should be max(f_best - mu, 0)
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)
    
    Z = (f_best - mu) / sigma_safe
    
    # Compute Phi(Z) and phi(Z)
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # EI for minimization: expected improvement over f_best
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Clamp EI to be non-negative (improvement can't be negative)
    EI = np.maximum(EI, 0.0)
    
    # Multiplicative cost discount: score = EI / cost
    # This explicitly maximizes expected reduction per unit of budget
    score = EI / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return score
    
    return score
