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
    Cost-aware acquisition: Probability of Significant Improvement (PSI)
    combined with a cost-normalized UCB exploration term.
    
    PSI uses a fixed epsilon threshold to focus on high-confidence improvements,
    while the UCB term (scaled by inverse cost) encourages exploration in
    uncertain, low-cost regions.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Significant Improvement (PSI)
    # Focus on improvements greater than epsilon over f_best
    # For minimization: P(f(x) < f_best - epsilon)
    # This is sharper than PI (epsilon=0) and focuses on meaningful gains
    epsilon = 0.1 * np.abs(f_best)  # Scale epsilon relative to f_best magnitude
    if np.abs(f_best) < 1e-6:
        epsilon = 0.1  # Fallback for near-zero f_best
    
    z_psi = (f_best - epsilon - mu) / sigma_safe
    psi = norm.cdf(z_psi)
    
    # Cost-relative normalization
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range  # [0, 1]
    else:
        cost_norm = np.zeros_like(cost)
    
    # UCB term for minimization: -mu + beta*sigma
    beta = 2.0
    ucb_term = -mu + beta * sigma_safe
    
    # Scale UCB by (1 - cost_norm) so high-cost candidates get less exploration bonus
    # This creates a smooth, relative cost penalty
    exploration = (1.0 - cost_norm) * ucb_term
    
    # Normalize exploration term to be comparable in scale to PSI [0,1]
    ucb_abs_max = np.abs(ucb_term).max()
    if ucb_abs_max > 1e-9:
        exploration_normalized = exploration / ucb_abs_max
    else:
        exploration_normalized = np.zeros_like(exploration)
    
    # Combine PSI (exploitation) with normalized exploration
    # PSI is in [0,1], exploration_normalized is roughly in [-1, 1]
    # Weight exploration lightly to not overwhelm the exploitation signal
    score = psi + 0.15 * exploration_normalized
    
    # Handle any NaNs or infs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
