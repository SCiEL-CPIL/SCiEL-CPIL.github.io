"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a weighted combination of:
    - Probability of Improvement (PI): robust, favors moderate improvements
    - Expected Improvement (EI): captures magnitude of improvement
    - Uncertainty bonus: encourages exploration in high-sigma regions
    
    The weights dynamically shift based on accumulated cost (budget pressure).
    As cost approaches the budget limit (30), we become more exploitative.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = evaluate next)
    """
    # Track elapsed time for safety margin
    t0 = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Normalize cost to [0, 1] range where 0 = start, 1 = budget limit
    # Budget limit is 30, so cost_fraction = cost / 30
    cost_fraction = np.clip(cost / 30.0, 0.0, 1.0)
    
    # Compute improvement-based metrics
    improvement = f_best - mu  # positive when mu < f_best (potential improvement)
    
    # Probability of Improvement (PI)
    z = improvement / sigma_safe
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI and PI to comparable scales
    # EI can be negative if mu > f_best, clip to 0
    ei_clipped = np.maximum(ei, 0.0)
    pi_clipped = np.clip(pi, 0.0, 1.0)
    
    # Uncertainty: high sigma means more to learn
    # Scale sigma to be comparable to improvement
    # Use a soft normalization: uncertainty_score = sigma / (sigma + typical_improvement)
    # Typical improvement scale: use median of positive improvements
    pos_improvements = improvement[improvement > 0]
    if len(pos_improvements) > 0:
        typical_improvement = np.median(pos_improvements)
    else:
        typical_improvement = 1e-3
    
    uncertainty_score = sigma_safe / (sigma_safe + typical_improvement)
    
    # Dynamic weighting based on cost pressure:
    # Early on (low cost): more exploration (higher weight on uncertainty)
    # Late on (high cost): more exploitation (higher weight on improvement)
    
    # Exploration weight decreases as cost increases
    # At cost=0: exploration_weight = 0.3, exploitation_weight = 0.7
    # At cost=30: exploration_weight = 0.05, exploitation_weight = 0.95
    exploration_weight = 0.3 * (1.0 - cost_fraction) + 0.05
    exploitation_weight = 1.0 - exploration_weight
    
    # Within exploitation, blend PI and EI
    # PI is more robust, EI captures magnitude
    # Use 70% PI, 30% EI for exploitation component
    pi_weight = 0.7
    ei_weight = 0.3
    
    # Normalize EI to [0, 1] range for blending with PI
    # Use a soft cap: ei_norm = 1 - exp(-ei / scale)
    ei_scale = typical_improvement + 1e-8
    ei_normalized = 1.0 - np.exp(-ei_clipped / ei_scale)
    
    # Combine
    exploitation_score = pi_weight * pi_clipped + ei_weight * ei_normalized
    total_score = exploitation_weight * exploitation_score + exploration_weight * uncertainty_score
    
    # Apply cost penalty: divide by cost to get per-unit-cost value
    # But keep it bounded to avoid numerical issues
    cost_safe = np.maximum(cost, 1e-8)
    score = total_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time budget - if we're running long, return immediately
    # (This is a simple guard; the computation above should be fast)
    elapsed = time.time() - t0
    if elapsed > 290:  # Safety margin before 300s limit
        # Return a simple fallback: just PI / cost
        z_fallback = (f_best - mu) / (sigma_safe + 1e-9)
        score = norm.cdf(z_fallback) / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
