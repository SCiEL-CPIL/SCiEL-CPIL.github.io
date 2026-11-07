"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Uses a cost-adjusted Expected Improvement (EI) combined with a
    budget-aware exploration bonus. The design shifts from exploration
    to exploitation as the effective budget depletes, estimated via
    the current f_best and the cost distribution.
    """
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Estimate progress: f_best decreasing from ~15-20 to 0
    # Use a robust progress metric
    f_ref = 20.0  # typical initial f_best for Powell 4D on [-4,5]^4
    progress = 1.0 - np.clip(f_best / f_ref, 0.0, 1.0)
    
    # Dynamic beta: high exploration early, low exploitation late
    beta = 0.3 + 1.2 * (1.0 - progress)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI: divide by cost to get improvement per unit cost
    # This naturally penalizes expensive evaluations
    cost_adj_ei = ei / cost_safe
    
    # Exploration bonus: uncertainty weighted by inverse cost
    # Encourages exploring cheap, uncertain regions
    exploration = beta * sigma_safe / cost_safe
    
    # Combine: weight shifts from exploration to exploitation
    w_ei = 0.4 + 0.6 * progress
    w_expl = 1.0 - w_ei
    
    # Normalize each component to comparable scales
    ei_std = np.std(cost_adj_ei) + 1e-8
    expl_std = np.std(exploration) + 1e-8
    
    score = w_ei * (cost_adj_ei / ei_std) + w_expl * (exploration / expl_std)
    
    # Time check: if we're running out of time, return immediately
    if time.time() - t0 > 0.5:
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
