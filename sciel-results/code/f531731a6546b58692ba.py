# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition for Bayesian optimization (minimization).
    
    score(x) = EI(x) / cost(x)
    
    where EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    This directly optimizes expected regret reduction per unit evaluation cost.
    """
    t_start = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.size

    # Numerical guards
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Standardized improvement: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -40.0, 40.0)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where phi is the standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical guard)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: maximize expected improvement per unit cost
    final_scores = ei / cost_safe

    # Ensure all values are finite (higher = evaluate next)
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)

    # Deadline guard (safety margin before the 330 s kill)
    if (time.time() - t_start) > 300.0:
        return final_scores

    return final_scores
# EVOLVE-BLOCK-END
