"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Normalized Expected Improvement (EI / cost).
    
    This approach directly optimizes for the expected improvement per unit of evaluation cost,
    aligning the search with the budget-constrained objective. Unlike additive penalties,
    division normalizes the improvement by the resource expenditure, creating a multiplicative
    trade-off that strongly favors efficient evaluations in low-cost regions while still
    allowing high-improvement candidates in high-cost regions to compete if their potential
    gain is sufficiently large.
    
    The deadline guard ensures the function returns within the 330s wall-clock limit.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin before 330s limit
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    n = mu.shape[0]
    
    # Ensure sigma is positive to avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm for cdf (Phi) and pdf (phi)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-Normalized EI: EI / cost
    # This score represents the expected improvement per unit cost.
    # Higher score = better candidate.
    score = ei / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
