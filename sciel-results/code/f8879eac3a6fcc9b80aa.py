"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the time limit, return a simple exploitation score
    if time.time() > _DEADLINE:
        # Simple exploitation: prefer points with lowest mu
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Cost-aware acquisition combining EI with uncertainty bonus
    # Use squared cost penalty to strongly penalize expensive evaluations
    # This aligns search with low-coordinate regions (lower cost)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus: reward exploration in uncertain regions
    # But discount by cost since expensive evaluations are less valuable
    uncertainty_bonus = sigma_safe / cost_safe
    
    # Combine: weighted EI with uncertainty bonus
    # Stronger cost penalty (power 2) helps find low-cost optima
    cost_penalty = cost_safe ** 2
    scores = ei / cost_penalty + 0.1 * uncertainty_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
