"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # For minimization, EI is the expected amount by which the new point
    # improves upon the current best f_best.
    # E[f(x) < f_best] * (f_best - E[f(x) | f(x) < f_best])
    # This simplifies to: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware adjustment
    # The knowledge suggests that dividing by cost**0.5 (square root) 
    # allows for better exploration of high-cost regions compared to linear cost.
    # This balances the trade-off between finding a low value and paying low cost.
    cost_penalty = np.sqrt(cost)
    
    # Score is EI per unit "sqrt" cost
    scores = ei / cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
