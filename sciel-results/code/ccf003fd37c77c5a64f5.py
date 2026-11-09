"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of Expected Improvement (EI)
    and Upper Confidence Bound (UCB), both normalized by cost.
    
    Structural change: Blends two distinct acquisition signals (EI for exploitation
    and UCB for exploration) in a cost-aware manner. This provides a more robust
    exploration-exploitation tradeoff than using EI alone, while still penalizing
    expensive evaluations.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple UCB / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        scores = (-mu + 2.0 * sigma_safe) / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement is f_best - f(x)
    # E[I] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Vectorized calculation
    z = (f_best - mu) / sigma
    
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB for minimization
    # UCB = f_best - mu + beta * sigma
    # This is the potential improvement if we were to evaluate x
    beta = 2.0
    ucb = f_best - mu + beta * sigma
    ucb = np.maximum(ucb, 0.0)  # Ensure non-negative
    
    # Cost-aware scores
    ei_cost = ei / cost
    ucb_cost = ucb / cost
    
    # Combine: weighted blend of cost-aware EI and cost-aware UCB
    # 0.7 for EI (exploitation), 0.3 for UCB (exploration)
    scores = 0.7 * ei_cost + 0.3 * ucb_cost
    
    # Handle non-finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
