"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines expected improvement (EI) with a cost-awareness mechanism.
    The key insight is that we want to maximize improvement per unit cost,
    but also need to balance exploitation vs exploration.
    
    We use a modified acquisition that:
    1. Computes expected improvement (EI)
    2. Scales by 1/cost to prefer cheaper evaluations
    3. Adds an exploration bonus weighted by uncertainty
    4. Uses a smooth, bounded formulation for numerical stability
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    if time.time() - _start_time > _DEADLINE:
        # Fallback: just return negative mu / cost (simple exploitation)
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        scores = (f_best - mu) / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.maximum(scores, 0.0)
        return scores
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to avoid extreme values
    ei = np.clip(ei, 0.0, None)
    
    # Cost-aware: improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # Exploration term: encourage visiting uncertain regions
    # Use a slightly larger coefficient and smoother saturation
    exploration = 0.01 * sigma_safe / (sigma_safe + 0.05)
    
    # Combine exploitation and exploration
    scores = ei_per_cost + exploration
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
