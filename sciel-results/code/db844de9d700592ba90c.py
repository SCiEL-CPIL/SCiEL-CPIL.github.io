"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    1. Early on (when f_best is not yet close to 0), use Probability of Improvement (PI)
       which is less biased towards high-variance points and can find minima faster.
    2. Later, use Expected Improvement (EI) per unit cost for cost-efficient exploitation.
    
    The cost-aware component divides by cost to naturally balance improvement vs. expense.
    A small exploration bonus based on uncertainty is added to encourage visiting
    under-explored regions.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Use a blend: PI for exploration, EI for exploitation
    # The blend weight can be adjusted based on how close we are to the optimum
    # When f_best is large (far from optimum), use more PI
    # When f_best is small (close to optimum), use more EI
    
    # Scale factor: as f_best approaches 0, shift towards EI
    # f_best is in objective units, griewank min is 0
    # Use a smooth transition
    # When f_best > 0.1, mostly PI; when f_best < 0.01, mostly EI
    blend = np.clip(1.0 - f_best / 0.1, 0.0, 1.0)
    blend = blend * 0.5 + 0.25  # Shift so we always have some EI component
    
    # Acquisition value before cost normalization
    acq = blend * ei + (1.0 - blend) * pi
    
    # Cost-aware: divide by cost to get improvement per unit cost
    acq_cost = acq / cost_safe
    
    # Add a small exploration bonus for high uncertainty
    # This helps escape local optima and explore under-explored regions
    # The bonus is scaled down as we get closer to the optimum
    exploration_bonus = 0.01 * sigma_safe * (1.0 + f_best)
    acq_cost += exploration_bonus / cost_safe
    
    # Ensure all values are finite
    acq_cost = np.where(np.isfinite(acq_cost), acq_cost, 0.0)
    
    return acq_cost
# EVOLVE-BLOCK-END
