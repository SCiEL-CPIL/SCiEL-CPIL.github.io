# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s/330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    Uses rank-based Pareto front selection between expected improvement and cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle NaN/Inf in inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    n = len(mu)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # UCB component for exploration
    beta = 2.0
    ucb = mu + beta * sigma
    ucb_improvement = f_best - ucb
    
    # Compute cost-normalized versions
    # Use sqrt(cost) for softer penalty than linear division
    cost_sqrt = np.sqrt(cost)
    
    ei_cost = ei / cost_sqrt
    ucb_cost = ucb_improvement / cost_sqrt
    pi_cost = pi / cost_sqrt
    
    # Rank-based Pareto front selection
    # Objective 1: maximize expected improvement (ei)
    # Objective 2: minimize cost
    # 
    # Compute non-dominated set and assign scores based on rank within front
    
    # First, compute a combined score for ranking
    # We want to balance ei (higher is better) and cost (lower is better)
    
    # Normalize ei and cost to [0, 1] for fair comparison
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    cost_min, cost_max = np.min(cost), np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range
    
    # Pareto dominance: candidate i dominates j if:
    # ei_i >= ei_j AND cost_i <= cost_j AND at least one is strict
    # 
    # For efficiency, we'll use a rank-based approach:
    # Sort by ei_norm descending, then assign ranks based on cost within that ordering
    # This creates a trade-off surface where high EI with low cost gets top ranks
    
    # Sort indices by ei_norm descending
    ei_sort_idx = np.argsort(-ei_norm)
    
    # Assign initial ranks based on ei ordering
    ranks = np.empty(n, dtype=np.float64)
    
    # For each candidate in ei-sorted order, check if it's dominated by any earlier candidate
    # A candidate is dominated if there exists an earlier candidate with lower or equal cost
    # (since earlier candidates have higher or equal ei)
    
    # Efficient approach: track minimum cost seen so far in ei-sorted order
    # If current cost > min_cost_so_far, it's dominated
    # If current cost <= min_cost_so_far, it's on the Pareto front
    
    min_cost_so_far = np.inf
    pareto_front = np.zeros(n, dtype=bool)
    
    for i in range(n):
        idx = ei_sort_idx[i]
        c = cost[idx]
        if c <= min_cost_so_far:
            pareto_front[idx] = True
            min_cost_so_far = c
    
    # For non-dominated points, assign high scores
    # For dominated points, assign lower scores based on how far they are from the front
    
    # Compute a "distance to Pareto front" metric for dominated points
    # Use the rank within the ei-sorted order: earlier = better
    
    # Assign base scores:
    # - Pareto front points get high base score
    # - Dominated points get lower scores based on their position
    
    # Create score array
    base_scores = np.zeros(n, dtype=np.float64)
    
    # Pareto front points: score proportional to ei_norm
    base_scores[pareto_front] = 1.0 + ei_norm[pareto_front]
    
    # Dominated points: score based on how much worse they are
    # Use a decay based on their ei rank position
    dominated_mask = ~pareto_front
    if np.any(dominated_mask):
        # For dominated points, assign scores based on ei_norm but with penalty
        # The penalty increases as the point is further from the front
        # Use the cost difference from the minimum cost on the front
        min_front_cost = min_cost_so_far if np.any(pareto_front) else cost_min
        cost_penalty = np.maximum(cost - min_front_cost, 0) / (cost_max - cost_min + 1e-9)
        base_scores[dominated_mask] = 0.5 * ei_norm[dominated_mask] - 0.3 * cost_penalty[dominated_mask]
    
    # Blend with the original cost-aware acquisition components
    # The Pareto front selection provides the structural trade-off
    # The EI/UCB/PI components provide refinement within that structure
    
    # Adaptive blending: weight the Pareto-based score more when there's clear structure
    # Weight the continuous acquisition more when points are close together
    
    # Compute a measure of how "spread out" the Pareto front is
    if np.sum(pareto_front) > 1:
        front_ei_spread = ei_max - ei_min  # simplified
        spread_weight = np.clip(front_ei_spread / (ei_range + 1e-9), 0.3, 0.7)
    else:
        spread_weight = 0.5
    
    # Continuous acquisition blend (from current program)
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    alpha = 0.3 + 0.4 * sigma_norm
    
    continuous_score = alpha * ucb_cost + (1 - alpha) * ei_cost
    continuous_score += 0.1 * pi_cost
    
    # Exploration bonus: Gaussian-like peak at moderate uncertainty
    sigma_center = 0.4
    sigma_width = 0.3
    exploration_bonus = 0.15 * np.exp(-0.5 * ((sigma_norm - sigma_center) / sigma_width) ** 2)
    
    # Final score: blend Pareto-based and continuous
    # Normalize both to comparable scales
    base_scores_max = np.max(base_scores) if np.max(base_scores) > 0 else 1.0
    base_scores_norm = base_scores / (base_scores_max + 1e-9)
    
    continuous_max = np.max(continuous_score) if np.max(continuous_score) > 0 else 1.0
    continuous_norm = continuous_score / (continuous_max + 1e-9)
    
    scores = spread_weight * base_scores_norm + (1 - spread_weight) * continuous_norm
    scores += exploration_bonus * 0.1  # small exploration bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
