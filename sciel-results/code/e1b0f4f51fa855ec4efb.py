"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Cost-normalized LCB with dynamic exploration
    and PI blending to avoid variance inflation in high dimensions.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # PI = Phi(z) - probability that f(x) < f_best
    pi = norm.cdf(z)
    
    # LCB: mu - kappa * sigma (lower is better, so we maximize -LCB)
    # Dynamic kappa: scale with remaining budget awareness
    # As budget depletes, we want to be more exploitative
    # Estimate remaining budget: total budget 30, current cost sum
    # But we don't know total evaluations done, so use a fixed reasonable kappa
    # In 8D, variance inflation is severe, so keep kappa moderate
    kappa = 1.5
    
    lcb = mu - kappa * sigma_safe
    # We want to minimize, so higher score = lower lcb
    lcb_score = -lcb
    
    # Blend LCB and PI: use a weighted combination
    # PI is in [0,1], LCB is in objective units
    # Normalize LCB to be comparable: scale by typical improvement range
    # Use f_best as reference; improvements are typically O(1) for cosine
    # Simple blend: 0.5 * normalized_lcb + 0.5 * pi
    # Normalize LCB by dividing by a scale factor
    lcb_scale = max(abs(f_best) + 1.0, 1.0)
    lcb_normalized = lcb_score / lcb_scale
    
    # Blend: weight toward PI for stability, LCB for exploitation
    blended = 0.6 * pi + 0.4 * lcb_normalized
    
    # Cost-aware score: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = blended / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
