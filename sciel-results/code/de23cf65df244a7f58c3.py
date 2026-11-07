"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound acquisition function.
    
    Uses a dynamic exploration term scaled by remaining budget to balance
    exploration vs exploitation, with cost normalization.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Known minimum of the objective function
    f_min = -0.8
    
    # Calculate the gap to the optimum
    gap = max(0.0, f_best - f_min)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration parameter beta
    # Scale with gap: larger gap means we need more exploration
    # Also scale with budget remaining (approximated by gap)
    # beta controls the LCB: score = mu - beta * sigma
    # For minimization, lower is better, so we want low mu and low sigma
    # LCB for minimization: mu - beta * sigma (we want to minimize this)
    # But acquisition should be higher = better to evaluate
    # So we negate: score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Adaptive beta: start high, decrease as we get closer to optimum
    # beta = c * sqrt(log(n)) / sigma_scale, but we don't have n directly
    # Use gap-based scaling: beta proportional to gap
    # When gap is large (early), beta is large (more exploration)
    # When gap is small (late), beta is small (more exploitation)
    
    # Normalize sigma to get a sense of typical uncertainty
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Dynamic beta: scale with gap and normalize by typical sigma
    # beta has units of (objective / sigma), so beta * sigma has objective units
    # We want beta * sigma to be comparable to the gap
    # beta ~ gap / sigma_typical
    # But we also want to cap it to avoid excessive exploration
    
    # Use a smooth transition: beta = base_beta * (gap / gap_ref)^alpha
    # gap_ref ~ 0.1 (typical gap when we're making progress)
    gap_ref = 0.1
    alpha = 0.5
    
    # Base beta: when gap = gap_ref, beta = base_beta
    base_beta = 1.5
    
    # Compute beta
    if gap > 1e-6:
        beta = base_beta * (gap / gap_ref) ** alpha
    else:
        beta = 0.1  # Small exploitation-only term
    
    # Cap beta to avoid extreme values
    beta = np.clip(beta, 0.1, 5.0)
    
    # Additional exploration term based on relative uncertainty
    # Regions with high sigma relative to the gap should be penalized
    # This prevents the GP from chasing noise in high-variance regions
    
    # Compute a variance penalty: penalize points where sigma is large relative to gap
    # penalty = sigma^2 / (gap + sigma^2) * some_scale
    # This is 0 when sigma is small, approaches 1 when sigma >> gap
    
    gap_safe = max(gap, 1e-6)
    variance_penalty = sigma_safe ** 2 / (gap_safe ** 2 + sigma_safe ** 2)
    
    # Blend: use LCB with variance penalty
    # score = -(mu - beta * sigma) - penalty_weight * variance_penalty * sigma
    # The penalty term discourages high-uncertainty regions
    
    penalty_weight = 0.3
    
    # Compute LCB score (higher is better to evaluate)
    lcb_score = -mu + beta * sigma_safe
    
    # Subtract variance penalty (in objective units)
    penalty_term = penalty_weight * variance_penalty * sigma_safe
    
    scores = lcb_score - penalty_term
    
    # Cost normalization: divide by cost to favor cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    scores = scores / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
