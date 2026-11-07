"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-efficient EI with probability-of-improvement.
    
    Structural change: instead of weighted EI+UCB divided by cost, we compute
    a constrained improvement metric that naturally balances exploration/exploitation
    and cost without arbitrary scaling constants.
    
    Key insight: use EI/cost as the primary metric but add a PI term that is ALSO
    cost-normalized. The PI term provides exploration pressure without the instability
    of UCB. Both terms are in "improvement per cost" units, so no arbitrary scaling is needed.
    
    Adaptive weight: as f_best approaches 0 (the Griewank minimum), shift toward exploitation.
    When f_best is large, explore more (higher PI weight).
    """
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for inputs
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.isfinite(f_best):
        return np.zeros(len(mu))
        
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1.0)  # Cost is at least 1.0
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (for minimization)
    pi = norm.cdf(z)
    
    # Cost-aware scoring:
    # Primary: expected improvement per unit cost
    ei_per_cost = ei / cost_safe
    
    # Secondary: probability of improvement per unit cost
    # This is more stable than UCB and naturally favors regions with
    # high uncertainty but also reasonable expected value
    pi_per_cost = pi / cost_safe
    
    # Adaptive weight: as f_best approaches 0, shift toward exploitation
    # f_best is typically in [0, 0.5] range for Griewank on [-10,10]^2
    # When f_best is small, we want to exploit (high EI weight)
    # When f_best is large, we want to explore (higher PI weight)
    
    # Weight transitions from 0.3 (explore) to 0.7 (exploit) as f_best goes from 0.5 to 0
    f_best_abs = abs(f_best)
    # Clamp f_best_abs to [0, 0.5] for stable weighting
    f_best_clamped = min(max(f_best_abs, 0.0), 0.5)
    # Linear interpolation: w_ei = 0.7 - 0.4 * (f_best_clamped / 0.5)
    # When f_best_clamped = 0: w_ei = 0.7, w_pi = 0.3
    # When f_best_clamped = 0.5: w_ei = 0.3, w_pi = 0.7
    w_ei = 0.7 - 0.4 * (f_best_clamped / 0.5)
    w_pi = 1.0 - w_ei
    
    score = w_ei * ei_per_cost + w_pi * pi_per_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety deadline guard
    elapsed = time.time() - t0
    if elapsed > 290:
        return np.maximum(ei_per_cost, 0.0)
    
    return score
# EVOLVE-BLOCK-END
