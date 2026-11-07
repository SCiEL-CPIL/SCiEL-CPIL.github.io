"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Rank-based Pareto front selection for cost-aware Bayesian optimization.
    
    Computes expected improvement (EI) for each candidate, then identifies the
    non-dominated set in the (EI, cost) space. Candidates on the Pareto front
    receive high scores proportional to their rank within the front, while
    dominated candidates receive lower scores based on their distance to the front.
    """
    # Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    n = len(ei)
    
    # Find non-dominated points in (ei, cost) space.
    # Point i is dominated if exists j: ei[j] > ei[i] and cost[j] <= cost[i], 
    # or ei[j] == ei[i] and cost[j] < cost[i].
    
    # Sort indices by ei descending, then cost ascending
    order = np.lexsort((cost, -ei))
    
    ei_sorted = ei[order]
    cost_sorted = cost[order]
    
    non_dominated_mask = np.zeros(n, dtype=bool)
    
    min_cost_so_far = np.inf
    i = 0
    while i < n:
        j = i
        while j < n and np.isclose(ei_sorted[j], ei_sorted[i], rtol=1e-12, atol=1e-15):
            j += 1
        
        min_cost_in_group = np.min(cost_sorted[i:j])
        idx_min_cost = np.argmin(cost_sorted[i:j]) + i
        
        if min_cost_in_group < min_cost_so_far:
            non_dominated_mask[order[idx_min_cost]] = True
            min_cost_so_far = min_cost_in_group
        
        i = j
    
    scores = np.zeros(n)
    
    front_indices = np.where(non_dominated_mask)[0]
    if len(front_indices) > 0:
        front_ei = ei[front_indices]
        front_cost = cost[front_indices]
        
        front_order = np.lexsort((front_cost, -front_ei))
        
        n_front = len(front_indices)
        for rank, idx_in_front in enumerate(front_order):
            idx = front_indices[idx_in_front]
            scores[idx] = 1.0 - (rank / max(n_front - 1, 1))
        
        scores[front_indices] = 1.0 + scores[front_indices]
    
    non_front_indices = np.where(~non_dominated_mask)[0]
    if len(non_front_indices) > 0:
        max_front_ei = np.max(ei[front_indices]) if len(front_indices) > 0 else np.max(ei)
        
        for idx in non_front_indices:
            ei_gap = max(0.0, max_front_ei - ei[idx])
            ei_scale = max(max_front_ei, 1e-9)
            penalty = ei_gap / ei_scale
            cost_penalty = cost[idx] / max(np.max(cost), 1e-9)
            score_val = max(0.0, 1.0 - 0.5 * penalty - 0.5 * cost_penalty)
            scores[idx] = score_val * 0.9
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
