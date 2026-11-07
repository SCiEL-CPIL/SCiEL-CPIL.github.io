"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Cost-Adjusted Expected Improvement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_EPS = 1e-12
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) acquisition function.
    
    Strategy:
    1. Use the analytic Expected Improvement (EI) for a Gaussian posterior.
       EI(x) = E[max(f_best - Y, 0)] where Y ~ N(mu, sigma^2).
       This is theoretically grounded for minimization.
    2. Adjust for cost: Divide EI by the local evaluation cost.
       This directly optimizes "improvement per unit cost", which is the
       correct objective when budget is constrained.
    3. No dynamic beta or heuristic penalties -- the cost adjustment is exact.
    
    The analytic EI for minimization:
      Let Z = (f_best - mu) / sigma  (standardized improvement)
      EI = sigma * [Z * Phi(Z) + phi(Z)]
      where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    Cost-aware score = EI / cost
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    t0 = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Standardized improvement: Z = (f_best - mu) / sigma
    # Since we minimize, improvement is f_best - Y, so mean improvement is f_best - mu
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in extreme tails
    # For z < -10, EI is negligible; for z > 10, EI ~ sigma * z
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Analytic EI: sigma * [Z * Phi(Z) + phi(Z)]
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = sigma * (z_clipped * Phi_z + phi_z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    score = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return score
        
    return score

# EVOLVE-BLOCK-END
