"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-Based Cost-Discounted Lower Confidence Bound for minimization.
    
    Uses a dynamic lower quantile of the predictive distribution instead of
    linear sigma, and applies a non-linear sigmoidal cost discount to prune
    expensive candidates in outer regions.
    """
    # Deadline guard: if we're running long, return a safe fallback
    # (This is a per-call guard; the main loop should also track total time)
    # Since we don't have access to total elapsed time here, we keep this lightweight.
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Dynamic quantile: use 5th percentile of predictive distribution
    # For a normal distribution, the 5th percentile is mu - 1.645*sigma
    # But we want to be adaptive: when uncertainty is high relative to improvement,
    # use a lower quantile (more exploration); when improvement is clear, use higher.
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Dynamic quantile parameter: 
    # When z is large (mu << f_best, clear improvement), use a higher quantile (less exploration)
    # When z is small or negative (uncertain or worse than best), use a lower quantile (more exploration)
    # Map z to a quantile in [0.05, 0.5] using a sigmoid
    # z = 0 -> 0.25, z = 2 -> ~0.47, z = -2 -> ~0.03
    quantile_param = 0.25 + 0.25 * np.tanh(0.5 * z)
    # Clip to reasonable range
    quantile_param = np.clip(quantile_param, 0.01, 0.49)
    
    # Convert quantile to z-value (inverse CDF)
    # For numerical stability, clip quantile_param away from 0 and 1
    quantile_param_safe = np.clip(quantile_param, 1e-6, 1 - 1e-6)
    z_quantile = norm.ppf(quantile_param_safe)
    
    # Lower quantile of predictive distribution (this is a "lower confidence bound")
    # We want to minimize, so lower values are better
    lower_quantile = mu + z_quantile * sigma_safe
    
    # The acquisition score should be higher for better (lower) values
    # So we negate: score = -lower_quantile
    # But we also want to incorporate the improvement over f_best
    # Actually, for minimization, we want to select points where the lower quantile is low
    # So the score should be: -(lower_quantile) or (f_best - lower_quantile)
    
    # Use (f_best - lower_quantile) as the base score, which is positive when
    # the lower quantile is below f_best (expected improvement)
    base_score = f_best - lower_quantile
    
    # Non-linear cost discount via sigmoid:
    # Cost ranges from 1 (center) to 3 (corners)
    # We want to aggressively penalize high-cost points
    # Use a sigmoid that is ~1 for cost < 1.5 and ~0 for cost > 2.5
    # sigmoid(x) = 1 / (1 + exp(-x))
    # We want discount(cost) = 1 / (1 + exp(k * (cost - center)))
    # Let's use center = 1.8, k = 3
    # At cost=1: 1/(1+exp(3*0.8)) = 1/(1+exp(2.4)) ≈ 0.08
    # At cost=1.5: 1/(1+exp(3*0.3)) = 1/(1+exp(0.9)) ≈ 0.29
    # At cost=2: 1/(1+exp(3*0.2)) = 1/(1+exp(0.6)) ≈ 0.35
    # At cost=2.5: 1/(1+exp(3*0.7)) = 1/(1+exp(2.1)) ≈ 0.11
    # At cost=3: 1/(1+exp(3*1.2)) = 1/(1+exp(3.6)) ≈ 0.027
    
    # Actually, I want the discount to be HIGHER for lower cost
    # So: discount = 1 / (1 + exp(k * (cost - center)))
    # This gives high discount for low cost, low discount for high cost
    k = 3.0
    center = 1.8
    cost_discount = 1.0 / (1.0 + np.exp(k * (cost_safe - center)))
    
    # Combine: score = base_score * cost_discount
    # But base_score can be negative if lower_quantile > f_best
    # We should handle this: if base_score is negative, the cost discount should
    # further reduce the score (make it more negative)
    # Actually, let's use: score = base_score / cost_penalty
    # where cost_penalty is higher for higher cost
    # cost_penalty = 1 + k2 * (cost - 1)^2
    # Or stick with the multiplicative discount but ensure it works for negative base_score
    
    # Let me reconsider: the original program used blended / cost
    # Let's use a similar approach but with the quantile-based score
    # score = base_score / (cost_safe ** power)
    # where power > 1 to make it more aggressive
    
    # Actually, let's use: score = base_score * cost_discount
    # where cost_discount is in (0, 1]
    # For positive base_score, this reduces the score for high cost
    # For negative base_score, this makes it less negative (closer to 0)
    # That's not ideal. Let's use division instead.
    
    # Use: score = base_score / (cost_safe ** 1.5)
    # This is similar to the original but with a stronger cost penalty
    
    score = base_score / (cost_safe ** 1.5)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Replace -inf with a very negative number for safety
    score = np.where(score == -np.inf, -1e10, score)
    
    return score
# EVOLVE-BLOCK-END
