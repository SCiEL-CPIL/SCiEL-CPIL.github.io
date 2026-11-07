"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Aware Expected Improvement (cEI).
    
    EI is theoretically grounded in maximizing the expected reduction in regret.
    By dividing EI by the evaluation cost, we explicitly optimize for 
    regret reduction per unit cost. This is more principled than ad-hoc 
    blending of UCB terms and handles the trade-off between finding a 
    slightly better point that is cheap vs. a much better point that is 
    expensive more effectively.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1.0)  # Cost is in [1, 2]

    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * pdf((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost to get improvement per unit cost
    score = ei / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
