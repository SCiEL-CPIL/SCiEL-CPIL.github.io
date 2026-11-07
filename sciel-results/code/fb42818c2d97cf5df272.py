"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a reward-space UCB with sqrt-cost scaling.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std dev predictions (positive)
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Estimate remaining budget ratio
    # Total budget is 30, initial 5 points cost ~5 (avg cost ~1.0)
    # So we start with cost ~5, budget remaining ~25
    # As cost accumulates, remaining decreases
    # We don't have explicit remaining budget, but we can infer from f_best progress
    # Actually, let's use a simpler approach: the cost array tells us the cost of each candidate
    # The total cost spent so far isn't directly available, but we can use f_best as a proxy
    
    # Better approach: use the cost values themselves to create a cost-aware score
    # Reward = expected improvement in objective = max(0, f_best - mu) approximately
    # But for UCB on reward: reward_ucb = (f_best - mu) + beta * sigma
    # This is high when mu is low (good prediction) and sigma is high (uncertainty)
    
    # Cost-aware: divide by sqrt(cost) to softly penalize expensive points
    # This is less aggressive than dividing by cost, allowing some exploration of expensive regions
    
    # Adaptive beta based on f_best:
    # When f_best is large (early), explore more (higher beta)
    # When f_best is small (late), exploit more (lower beta)
    
    # Three hump camel range: min=0, max can be ~4-5 in [-5,5]^2
    # Let's map f_best to beta
    f_best_safe = max(f_best, 1e-10)
    
    # Use a smooth decay: beta = beta_max * exp(-f_best / scale)
    # beta_max = 2.0, scale = 1.0
    # When f_best=0, beta=2.0; when f_best=1, beta=0.74; when f_best=3, beta=0.22
    beta = 2.0 * np.exp(-f_best_safe / 1.0)
    
    # Also add a floor so we never have zero exploration
    beta = max(beta, 0.1)
    
    # Reward-space UCB: (f_best - mu) + beta * sigma
    # This is the expected improvement plus uncertainty bonus
    reward_ucb = (f_best - mu) + beta * sigma_safe
    
    # Cost-aware scaling: divide by sqrt(cost)
    # cost is in [1, 2] for this problem (1 + mean(x_scaled))
    cost_safe = np.maximum(cost, 1e-9)
    score = reward_ucb / np.sqrt(cost_safe)
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip extreme values for numerical stability
    score_max = np.max(score)
    score_min = np.min(score)
    if not np.isfinite(score_max) or not np.isfinite(score_min):
        return np.zeros_like(score)
    
    # Normalize to [0, 1] for stability
    range_score = score_max - score_min
    if range_score > 1e-12:
        score = (score - score_min) / range_score
    else:
        score = np.zeros_like(score)
    
    return score
# EVOLVE-BLOCK-END
