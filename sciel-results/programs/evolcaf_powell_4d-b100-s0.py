"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural change: Replace per-candidate cost normalization with a global
    budget-aware dynamic that shifts from exploration to exploitation as the
    budget depletes. Uses a hybrid of Expected Improvement and a budget-scaled
    uncertainty bonus.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Budget-aware dynamic: estimate remaining budget ratio
    # Total budget is 30, initial cost from 5 uniform points is roughly 5 * 1.5 = 7.5
    # Current total cost is sum of all evaluations so far
    # We estimate the current iteration by looking at the cost array
    # The cost array contains the cost of each candidate, not cumulative
    # We need to estimate how much budget has been used
    
    # A simpler approach: use the mean cost as a proxy for budget depletion
    # Early: mean cost ~ 1.5, Late: mean cost could be higher if we're evaluating expensive points
    # But actually, the cost is per-evaluation, so we can't directly get total spent from this array
    
    # Alternative: Use a fixed schedule based on the distribution of mu values
    # When f_best is still high relative to the range, we're early; when it's low, we're late
    
    # For Powell 4D on [-4,5]^4, the minimum is 0
    # f_best starts around 10-20 and decreases to 0
    
    # Use a normalized progress metric
    # Estimate the range of f values: Powell 4D max is around 20-30
    f_range = max(abs(f_best), 1.0) + 1.0
    progress = 1.0 - (f_best / (f_range * 0.5))
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic beta: high exploration early, low exploitation late
    beta = 0.5 + 1.5 * (1.0 - progress)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus scaled by budget-aware beta
    # Higher beta = more exploration (uncertainty bonus)
    # Lower beta = more exploitation (EI)
    exploration = beta * sigma_safe
    
    # Combine EI and exploration
    # Normalize both to comparable scales
    ei_scale = np.std(ei) + 1e-8
    expl_scale = np.std(exploration) + 1e-8
    
    ei_norm = ei / ei_scale
    expl_norm = exploration / expl_scale
    
    # Weight: shift from exploration to exploitation as budget depletes
    w_ei = 0.3 + 0.7 * progress
    w_expl = 1.0 - w_ei
    
    combined = w_ei * ei_norm + w_expl * expl_norm
    
    # Global cost penalty: subtract a term proportional to cost
    # This is a global penalty, not per-candidate normalization
    # Use a soft penalty that grows with cost
    cost_penalty = 0.3 * np.log1p(cost_safe)
    
    score = combined - cost_penalty
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
