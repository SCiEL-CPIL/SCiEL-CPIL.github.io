"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any NaN or Inf values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is strictly positive for division
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is within reasonable bounds for Rastrigin task
    # Cost is 1 + mean(x_scaled), so [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Compute cost-effective improvement: EI per unit cost
    # This directly optimizes for improvement relative to evaluation cost
    cost_effective_ei = ei / cost
    
    # Also compute a simple regret proxy: how much better could this be vs f_best
    # weighted by confidence (lower sigma = more confident)
    confidence_weight = 1.0 / (sigma + 1e-6)
    regret_proxy = (f_best - mu) * confidence_weight
    regret_proxy = np.maximum(regret_proxy, 0.0)
    
    # Normalize both metrics
    ce_max = np.max(cost_effective_ei)
    if ce_max > 1e-12:
        ce_norm = cost_effective_ei / ce_max
    else:
        ce_norm = np.zeros_like(cost_effective_ei)
    
    rp_max = np.max(regret_proxy)
    if rp_max > 1e-12:
        rp_norm = regret_proxy / rp_max
    else:
        rp_norm = np.zeros_like(regret_proxy)
    
    # Combine the two normalized metrics
    combined = 0.5 * ce_norm + 0.5 * rp_norm
    
    # Rank-based Pareto front selection:
    # Compute rank of each candidate by combined score (higher = better)
    n = len(combined)
    if n > 1:
        score_rank = np.argsort(np.argsort(combined)) / (n - 1)
    else:
        score_rank = np.ones_like(combined)
    
    # Compute rank of each candidate by cost (lower cost = higher rank)
    if n > 1:
        cost_rank = np.argsort(np.argsort(-cost)) / (n - 1)
    else:
        cost_rank = np.ones_like(cost)
    
    # True Pareto front detection:
    # A candidate is non-dominated if no other candidate has both
    # higher combined score AND lower cost
    # Efficient O(n log n) approach using sorting
    
    # Sort by cost ascending, then for each position check if any earlier 
    # (lower cost) candidate has higher combined score
    cost_order = np.argsort(cost)
    sorted_cost = cost[cost_order]
    sorted_combined = combined[cost_order]
    
    # Running max of combined score from left (lower cost side)
    # A point is dominated if there exists a point with lower cost AND higher combined score
    running_max = np.zeros(n)
    running_max[0] = sorted_combined[0]
    for i in range(1, n):
        running_max[i] = max(running_max[i-1], sorted_combined[i])
    
    # A point at position i is non-dominated if its combined score >= running max before it
    # (i.e., no point with strictly lower cost has strictly higher combined score)
    # Actually, we need: no point j with cost[j] < cost[i] AND combined[j] > combined[i]
    # So point i is dominated if running_max[i-1] > sorted_combined[i] (for i>0)
    # For i=0, it's always non-dominated (lowest cost)
    
    is_dominated = np.zeros(n, dtype=bool)
    if n > 1:
        # For each position, check if any previous position has higher combined score
        # running_max[i-1] is the max combined score among positions 0..i-1
        is_dominated[1:] = running_max[:-1] > sorted_combined[1:]
    
    # Map back to original indices
    non_dominated_mask = np.zeros(n, dtype=bool)
    non_dominated_mask[cost_order[~is_dominated]] = True
    
    # Assign higher scores to non-dominated candidates
    # Among non-dominated, rank by combined score
    # Among dominated, rank lower
    
    final_score = np.zeros(n)
    if np.any(non_dominated_mask):
        nd_indices = np.where(non_dominated_mask)[0]
        # Rank non-dominated by their combined score
        nd_combined = combined[nd_indices]
        if len(nd_combined) > 1:
            nd_rank = np.argsort(np.argsort(nd_combined)) / (len(nd_combined) - 1)
        else:
            nd_rank = np.ones_like(nd_combined)
        final_score[nd_indices] = 0.6 + 0.4 * nd_rank  # [0.6, 1.0]
    
    if np.any(~non_dominated_mask):
        dom_indices = np.where(~non_dominated_mask)[0]
        # Rank dominated by their combined score, but give lower scores
        dom_combined = combined[dom_indices]
        if len(dom_combined) > 1:
            dom_rank = np.argsort(np.argsort(dom_combined)) / (len(dom_combined) - 1)
        else:
            dom_rank = np.ones_like(dom_combined)
        final_score[dom_indices] = 0.3 * dom_rank  # [0, 0.3]
    
    # Ensure finite output
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return final_score
# EVOLVE-BLOCK-END
