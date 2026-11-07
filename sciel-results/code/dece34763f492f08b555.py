"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement per unit cost (EI/cost) as the primary score,
    which directly optimizes marginal return per budget spent.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) - in objective units
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Primary: EI per unit cost (BOB - Bayesian Optimization under Budget)
    # This directly maximizes expected improvement per unit of budget
    ei_per_cost = ei / cost_safe
    
    # Secondary: small UCB exploration term to avoid premature convergence
    # Use a modest exploration coefficient
    improvement = np.maximum(f_best - mu, 0)
    beta = 0.3 * np.exp(-improvement / (sigma_safe + 1e-9))
    beta = np.clip(beta, 0.05, 0.5)
    
    ucb = -mu + beta * sigma_safe
    ucb_per_cost = ucb / cost_safe
    
    # Combine: primarily EI/cost with small exploration component
    # Weight 0.9 for EI/cost, 0.1 for UCB/cost
    score = 0.9 * ei_per_cost + 0.1 * ucb_per_cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    if time.time() - t0 > 280:
        # Return simple cost-normalized EI as fallback
        fallback = ei / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
# EVOLVE-BLOCK-END
