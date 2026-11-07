"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 295.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines expected improvement (EI) with a cost-awareness mechanism and
    a smooth exploration bonus. The key insight is to maximize improvement
    per unit cost while balancing exploitation vs exploration.
    
    We use a modified acquisition that:
    1. Computes expected improvement (EI)
    2. Scales by 1/cost to prefer cheaper evaluations
    3. Adds a bounded, smooth exploration bonus weighted by uncertainty
    4. Uses numerically stable formulations
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
    # Bounded smooth term that scales with uncertainty but saturates
    # The denominator (sigma + 0.1) keeps the bonus O(1) and makes it
    # negligible when sigma is small
    exploration = 0.05 * sigma_safe / (sigma_safe + 0.1)
    
    # Combine exploitation and exploration
    scores = ei_per_cost + exploration
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
