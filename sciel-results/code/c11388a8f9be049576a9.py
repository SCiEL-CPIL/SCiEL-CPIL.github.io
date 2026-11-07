# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) for Bayesian Optimization.
    
    Strategy:
    1. Compute Expected Improvement (EI) using the standard formula for minimization:
       EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
       where Phi is the CDF and phi is the PDF of the standard normal distribution.
    2. This metric naturally balances exploration (high sigma) and exploitation (low mu)
       relative to the current best observed value (f_best). Unlike UCB, EI is bounded 
       and directly estimates the expected reduction in the objective function.
    3. Apply a cost-aware penalty by dividing the EI by the evaluation cost.
       This implements a "return on investment" metric: it favors points that offer 
       high expected improvement relative to their cost. Since cost is 1 + mean(x_scaled),
       points near the center (low cost) are favored if their EI is comparable to 
       edge points (high cost).
    4. This structural change moves away from heuristic exploration coefficients (beta) 
       or complex local kernel approximations to a principled probabilistic bound that 
       has proven effective in cost-aware BO literature.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero in z-score calculation
    # GP sigma should be > 0, but numerical issues can occur
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Z-score for EI
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, meaning we expect improvement.
    # If mu > f_best, z < 0, meaning we expect no improvement (EI will be 0 due to Phi(z) ~ 0).
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI should be non-negative. Due to floating point errors, small negative values might occur.
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted EI
    # We want to maximize EI per unit cost.
    ca_ei = ei / cost
    
    # Ensure finite values
    ca_ei = np.where(np.isfinite(ca_ei), ca_ei, 0.0)
    
    return ca_ei
# EVOLVE-BLOCK-END
