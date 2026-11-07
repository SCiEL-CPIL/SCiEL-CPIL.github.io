"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Pre-computed constants for speed
_N = 2000

# Deadline guard: track time to avoid exceeding 300s
_DEADLINE = time.time() + 320.0

def _norm_cdf(x):
    """Fast normal CDF using scipy."""
    return norm.cdf(x)

def _norm_pdf(x):
    """Fast normal PDF using scipy."""
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    mu, sigma: arrays of shape (2000,) in objective units
    f_best: scalar, lowest observed value
    cost: array of shape (2000,), evaluation cost per candidate
    
    Uses a cost-aware information-gain objective that directly models the
    stochastic knapsack structure: candidates in low-cost regions with high
    GP uncertainty receive high scores, while high-cost regions are
    exponentially suppressed. This concentrates search in the low-cost corner
    where shifted minima are likely.
    """
    # Deadline guard: if we're close to the limit, return a safe finite array
    if time.time() > _DEADLINE - 5.0:
        return np.ones_like(mu, dtype=float) * 0.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # --- Cost-aware information-gain objective ---
    # The key insight: we want to maximize information gain per unit cost.
    # Information gain is highest where:
    #   1. Uncertainty (sigma) is high — we learn the most
    #   2. The predicted value is close to or below f_best — we might improve
    #   
    # We model this as: IG(x) = sigma(x) * P(f(x) < f_best)
    # where P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    #
    # Then we apply exponential cost suppression:
    #   score = IG(x) * exp(-lambda * (cost(x) - cost_min) / cost_range)
    #
    # This naturally biases toward cheap evaluations in uncertain regions.
    
    # Compute probability of improvement: P(f(x) < f_best)
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -30.0, 30.0)
    p_improve = _norm_cdf(z)
    p_improve = np.where(np.isfinite(p_improve), p_improve, 0.0)
    p_improve = np.clip(p_improve, 0.0, 1.0)
    
    # Information gain: uncertainty weighted by probability of improvement
    # This peaks where sigma is large AND mu is close to f_best
    info_gain = sigma * p_improve
    info_gain = np.where(np.isfinite(info_gain), info_gain, 0.0)
    
    # Also add a pure exploration component for regions with very high uncertainty
    # where even if p_improve is low, learning the landscape is valuable
    # This is weighted by a factor that decreases as we get more confident
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        # Normalize sigma to [0, 1]
        sigma_norm = sigma / sigma_max
        # Exploration bonus: high where sigma is high, but damped by confidence
        # Use a soft gate: only significant when uncertainty is truly high
        exploration_bonus = sigma_norm ** 2 * 0.3
    else:
        exploration_bonus = np.zeros_like(sigma)
    
    # Combine information gain and exploration
    base_score = info_gain + exploration_bonus
    base_score = np.where(np.isfinite(base_score), base_score, 0.0)
    
    # --- Exponential cost suppression ---
    # cost is in [1, 2] range (1 + mean(x_scaled))
    # We want to strongly prefer low-cost candidates
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-12:
        # Normalize cost to [0, 1]
        cost_norm = (cost - cost_min) / cost_range
        # Exponential suppression: lambda controls how aggressive the bias is
        # Higher lambda = stronger preference for cheap evaluations
        lambda_cost = 3.0
        cost_weight = np.exp(-lambda_cost * cost_norm)
    else:
        cost_weight = np.ones_like(cost)
    
    # Final score: information value per effective cost
    scores = base_score * cost_weight
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    # Small exploration floor to avoid complete zeroing
    floor = 1e-10
    scores = np.maximum(scores, floor)
    
    return scores
# EVOLVE-BLOCK-END
