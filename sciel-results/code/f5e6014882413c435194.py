"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    - Early phase: aggressive exploration via UCB with high beta
    - Late phase: exploitation via Expected Improvement (EI)
    - Cost-awareness: normalize by cost to prefer cheaper evaluations
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        GP mean predictions
    sigma : ndarray, shape (n_candidates,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far (lower is better)
    cost : ndarray, shape (n_candidates,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute total cost consumed so far (approximate from cost array)
    # The cost array represents the cost of each candidate, but we need
    # to estimate how much budget has been used. Since we don't have direct
    # access to total_cost, we'll use a heuristic based on the number of
    # candidates and typical costs.
    
    # Actually, looking at the task description more carefully:
    # "While total cost is below 30" - we need to track this
    # But the function signature only gives us mu, sigma, f_best, cost
    # 
    # Let's use a simpler approach: use the average cost as a proxy for
    # budget consumption. If average cost is high, we're later in the process.
    
    mean_cost = np.mean(cost)
    
    # Adaptive beta for UCB: start high (exploration), decrease as budget depletes
    # Budget is 30, initial cost is ~5 (5 uniform points with cost ~1.5 each = 7.5)
    # So remaining budget ranges from ~22.5 to 0
    # Let's estimate remaining fraction
    # Assume total cost will be around 30
    # If we've spent C, remaining = 30 - C
    # We can estimate C from the fact that we're being called repeatedly
    
    # A simpler heuristic: use the minimum cost in the current batch as a proxy
    # for how "early" we are. Early on, we'll see many low-cost candidates.
    # Actually, let's just use a fixed schedule based on typical usage patterns.
    
    # For a 3D problem with cost 1 + mean(x), average cost is ~1.5
    # 5 initial points cost ~7.5
    # Each subsequent point costs ~1.5
    # So after k evaluations, total cost ≈ 7.5 + 1.5*(k-5) = 1.5*k
    # Budget 30 means about 20 evaluations total
    # We don't know k, but we can use the fact that sigma decreases over time
    
    # Use a smooth transition based on the uncertainty level
    # When sigma is large (early), explore more; when small (late), exploit more
    median_sigma = np.median(sigma_safe)
    
    # Normalize sigma: early on, sigma might be ~0.5-1.0; late, ~0.01-0.1
    # Use a logistic-like transition
    sigma_scale = 0.1  # threshold
    exploration_weight = 1.0 / (1.0 + np.exp(10 * (median_sigma - sigma_scale)))
    
    # Beta for UCB: high when exploring, low when exploiting
    beta = 0.5 + 2.0 * exploration_weight  # range [0.5, 2.5]
    
    # Compute UCB scores
    ucb_scores = mu + beta * sigma_safe
    
    # Compute EI scores
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend UCB and EI based on exploration weight
    # Early: more UCB; Late: more EI
    blended = exploration_weight * ucb_scores + (1.0 - exploration_weight) * ei_scores
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # But we want to be careful: dividing by cost can over-penalize expensive
    # points that might be much better. Use a softer normalization.
    cost_safe = np.maximum(cost, 1e-9)
    # Use log cost for softer penalty
    cost_penalty = np.log(cost_safe)
    
    # Final score: blend minus cost penalty
    scores = blended - 0.1 * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
