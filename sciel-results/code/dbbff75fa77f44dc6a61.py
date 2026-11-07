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
    Cost-aware acquisition using Probability of Improvement (PI) with
    relative-cost-normalized exploration. Structural change from EI/UCB hybrid
    to PI + cost-relative UCB, which provides a sharper exploitation signal
    and a more adaptive exploration penalty.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (for minimization)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Cost-relative UCB: normalize cost so exploration penalty adapts to
    # the cost distribution of the candidate pool
    cost_min = np.min(cost)
    cost_range = np.max(cost) - cost_min
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range  # [0, 1]
    else:
        cost_norm = np.zeros_like(cost)
    
    # UCB term for minimization: -mu + beta*sigma
    beta = 2.0
    ucb_term = -mu + beta * sigma_safe
    
    # Scale UCB by (1 - cost_norm) so high-cost candidates get less exploration bonus
    # This creates a smooth, relative cost penalty rather than absolute division
    exploration = (1.0 - cost_norm) * ucb_term
    
    # Combine: PI provides sharp exploitation, scaled exploration provides
    # cost-aware uncertainty bonus
    # Weight: PI is [0,1], ucb_term can be large; balance with scaling
    score = pi + 0.15 * exploration / (np.abs(ucb_term).max() + 1e-9)
    
    # Handle any NaNs or infs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
