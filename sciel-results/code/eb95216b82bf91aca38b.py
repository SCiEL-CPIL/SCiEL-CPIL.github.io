"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement per Unit Cost with
    relative-cost-normalized UCB exploration.
    
    EI provides a strong exploitation signal (probability + magnitude of improvement),
    divided by cost to penalize expensive evaluations. UCB exploration is scaled by
    relative cost to adaptively reduce exploration bonus for high-cost candidates.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-relative normalization for exploration term
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range  # [0, 1]
    else:
        cost_norm = np.zeros_like(cost)
    
    # UCB term for minimization: -mu + beta*sigma
    beta = 1.5
    ucb_term = -mu + beta * sigma_safe
    
    # Scale UCB by (1 - cost_norm) so high-cost candidates get less exploration bonus
    exploration = (1.0 - cost_norm) * ucb_term
    
    # Normalize exploration term to be comparable in scale to EI/cost
    ucb_abs_max = np.abs(ucb_term).max()
    if ucb_abs_max > 1e-9:
        exploration_normalized = exploration / ucb_abs_max
    else:
        exploration_normalized = np.zeros_like(exploration)
    
    # EI per unit cost: primary exploitation signal
    ei_per_cost = ei / (cost + 1e-9)
    
    # Combine: EI/cost dominates, exploration adds adaptive uncertainty bonus
    # Weight exploration lightly to not overwhelm the EI signal
    score = ei_per_cost + 0.1 * exploration_normalized
    
    # Handle any NaNs or infs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
