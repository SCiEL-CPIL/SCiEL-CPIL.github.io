"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 300s as safety margin
_DEADLINE_START = time.time()
_TIME_LIMIT = 300.0  # seconds, leave 30s margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware knowledge-gradient acquisition function.
    
    The KG approximates the expected reduction in the minimum observed value
    if we evaluate at each candidate, normalized by evaluation cost.
    This prioritizes high-value, low-cost regions.
    
    KG(x_i) ≈ E[max(0, f_best - f(x_i))] / cost(x_i)
    
    We add a small UCB exploration bonus to encourage visiting uncertain regions,
    with the weight adapting to how much improvement is still possible.
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Safe fallback: simple cost-normalized EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        ei_cost = ei / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(ei_cost), ei_cost, 0.0)
        return score
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Knowledge-Gradient: expected improvement in best value
    # KG = E[max(0, f_best - f(x))] = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized KG
    cost_safe = np.maximum(cost, 1e-9)
    kg_cost = ei / cost_safe
    
    # Small UCB exploration bonus to visit uncertain regions
    # Weight it lightly so exploitation dominates
    # Adapt: more exploration early (when f_best is far from min(mu))
    mu_min = np.min(mu)
    mu_range = np.max(mu) - mu_min + 1e-9
    f_best_norm = (f_best - mu_min) / mu_range
    
    # Exploration weight: small, increases when f_best is high relative to mu range
    # Cap at 0.3 to keep exploitation dominant
    beta = 0.3 * f_best_norm
    ucb_cost = (beta * sigma) / cost_safe
    
    # Combine: KG dominates, UCB provides gentle exploration
    score = kg_cost + ucb_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
