"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of the
    multiple-constraint knapsack problem.
    
    Uses a combination of:
    1. Per-constraint prize/weight ratio (efficiency)
    2. Soft-min across constraints to penalize items poor on any constraint
    3. Contribution-based scoring via greedy pack analysis
    4. Regret-based penalty for items that block high-value combinations
    """
    t0 = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Ensure positive values
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    # Per-constraint efficiency ratios
    ratios = prizes / weights  # shape (n_constraints, n_items)
    
    # Soft-min across constraints: penalize items that are poor on any constraint
    # softmin(x) = -logsumexp(-x) approximates min(x) smoothly
    # Use a temperature parameter for the soft-min
    temp = 0.1
    neg_ratios = -ratios / temp
    max_neg = np.max(neg_ratios, axis=0, keepdims=True)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_ratios - max_neg), axis=0, keepdims=True))
    soft_min_ratio = -log_sum_exp * temp  # shape (1, n_items)
    
    # Base heuristic: geometric mean of ratios, adjusted by soft-min penalty
    geo_mean_ratio = np.exp(np.mean(np.log(ratios), axis=0))
    
    # Contribution-based scoring: for each item, estimate its marginal contribution
    # to a greedy pack
    capacity = 25.0
    contributions = np.zeros(n_items)
    
    # Greedy pack by ratio (per constraint, averaged)
    avg_ratio = np.mean(ratios, axis=0)
    order = np.argsort(-avg_ratio)
    
    # Build a reference greedy pack
    remaining = np.full(n_constraints, capacity)
    pack_indices = []
    for idx in order:
        if np.all(weights[:, idx] <= remaining):
            pack_indices.append(idx)
            remaining -= weights[idx]
    
    # Compute contribution of each packed item: prize if included vs. best alternative
    # Simplified: contribution = prize * (how well it fits relative to alternatives)
    for idx in pack_indices:
        # How much slack does this item use?
        usage = weights[idx] / capacity
        # Contribution weighted by how efficiently it uses capacity
        contributions[idx] = prizes[:, idx] * (1.0 - np.mean(usage))
    
    # Regret-based penalty: items whose exclusion barely changes greedy pack value
    # are likely blocking high-value combinations
    # Approximate: for each item, compute how much the pack value drops if we remove it
    regret = np.zeros(n_items)
    if len(pack_indices) > 1:
        for i, idx in enumerate(pack_indices):
            # Value of pack without this item
            pack_without = [p for p in pack_indices if p != idx]
            val_with = np.sum(prizes[:, idx])
            # Items that are hard to replace have high regret
            # Approximate by the ratio of its prize to the average prize of other packed items
            if len(pack_without) > 0:
                avg_other = np.mean([np.sum(prizes[:, p]) for p in pack_without])
                regret[idx] = val_with / (avg_other + 1e-9)
    
    # Combine signals
    # Normalize each component to [0, 1]
    def normalize(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < 1e-12:
            return np.ones_like(x)
        return (x - x_min) / (x_max - x_min)
    
    norm_geo = normalize(geo_mean_ratio)
    norm_softmin = normalize(soft_min_ratio.ravel())
    norm_contrib = normalize(contributions)
    norm_regret = normalize(regret)
    
    # Weighted combination
    # Higher weight on efficiency, moderate on contribution, some on regret
    score = (0.4 * norm_geo + 0.3 * norm_softmin + 0.2 * norm_contrib + 0.1 * norm_regret)
    
    # Scale to ensure positive values with good dynamic range
    score = score + 0.1
    score = score / np.max(score)  # Normalize to max 1.0
    
    # Ensure all values are positive and finite
    score = np.where(np.isfinite(score) & (score > 0), score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
