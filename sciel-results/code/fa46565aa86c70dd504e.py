# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) acquisition function.
    
    Strategy:
    - Use LCB for minimization: LCB = mu - beta * sigma.
    - We want to maximize the probability of improvement while considering cost.
    - The score is derived from the expected improvement under the LCB framework,
      adjusted by the evaluation cost.
    - Specifically, we compute a cost-aware score that balances the potential
      improvement (f_best - mu) against the uncertainty (sigma) and the cost.
    
    Structural Change:
    - Switch from EI to a LCB-based score.
    - The score is: (f_best - mu) / (1 + cost) + beta * sigma / (1 + cost)
    - This simplifies to: ((f_best - mu) + beta * sigma) / (1 + cost)
    - This encourages finding points with low mu (high potential improvement)
      and high sigma (uncertainty), while penalizing high cost.
    """
    # Deadline guard
    if time.time() > _start_time + 300.0 - _DEADLINE_MARGIN:
        # Return a fallback score to avoid timeout
        mu_arr = np.asarray(mu)
        cost_arr = np.asarray(cost)
        return -mu_arr / (1.0 + cost_arr + 1e-9)
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-8)
    
    # LCB-based score for minimization
    # We want to maximize: (f_best - mu) + beta * sigma, all divided by cost
    # beta controls the exploration/exploitation trade-off
    beta = 2.0
    
    # Numerator: potential improvement + uncertainty bonus
    numerator = (f_best - mu) + beta * sigma
    
    # Denominator: cost + 1 to ensure cost is at least 1
    denom = 1.0 + cost
    denom = np.maximum(denom, 1e-9)
    
    scores = numerator / denom
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
