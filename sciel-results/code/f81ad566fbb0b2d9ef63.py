"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 290.0  # Safety margin before 330s kill


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines three complementary efficiency metrics with adaptive scarcity weighting:
    1. Aggregate Efficiency: Total Prize / Total Weight (overall density).
    2. Scarcity-Weighted Geometric Mean: Emphasizes efficiency in tighter constraints.
    3. Constraint-Specific Scarcity Premium: Identifies the most binding constraint
       per item and scales by inverse remaining capacity fraction.
    
    All metrics are normalized to mean 1 and combined with learned weights.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    eps = 1e-9
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Total prize per item (sum across constraints)
    item_prizes = prizes.sum(axis=0)
    
    # 1. Aggregate Efficiency: Prize / Sum(Weights)
    total_weights = weights.sum(axis=0)
    agg_ratio = item_prizes / (total_weights + eps)
    
    # 2. Per-constraint efficiency: Prize_c / Weight_c
    # Use mean prize per constraint for fair per-constraint comparison
    mean_prize_per_constraint = prizes.mean(axis=0)  # (100,)
    eff = mean_prize_per_constraint[None, :] / (weights + eps)  # (5, 100)
    eff = np.where(np.isfinite(eff), eff, eps)
    
    # 3. Scarcity-based constraint weighting:
    # Constraints with less total weight are tighter (scarcer)
    weight_sum = weights.sum(axis=1)  # (5,)
    scarcity = 1.0 / (weight_sum + eps)  # (5,)
    scarcity = scarcity / (scarcity.sum() + eps)  # normalize to sum to 1
    
    # Scarcity-weighted geometric mean in log-space
    log_eff = np.log(eff + eps)  # (5, 100)
    weighted_log = (scarcity[:, None] * log_eff).sum(axis=0)  # (100,)
    scarcity_geo = np.exp(weighted_log)  # (100,)
    
    # 4. Constraint-Specific Scarcity Premium:
    # For each item, find the constraint where its weight is the largest fraction
    # of total weight, then scale by inverse of remaining capacity for that constraint
    # weight_fraction[c, i] = weights[c, i] / total_weights[i]
    weight_fraction = weights / (total_weights[None, :] + eps)  # (5, 100)
    
    # Most binding constraint for each item
    most_binding = np.argmax(weight_fraction, axis=0)  # (100,)
    
    # Remaining capacity fraction for the most binding constraint
    # remaining_frac[c, i] = (capacity - weights[c, i]) / capacity
    remaining_frac = (capacity - weights) / (capacity + eps)  # (5, 100)
    
    # Get remaining fraction for the most binding constraint of each item
    item_indices = np.arange(n_items)
    binding_remaining = remaining_frac[most_binding, item_indices]  # (100,)
    
    # Scarcity premium: inverse of remaining fraction (higher when less capacity remains)
    # Clip to avoid division by zero
    binding_remaining = np.maximum(binding_remaining, eps)
    scarcity_premium = 1.0 / binding_remaining  # (100,)
    
    # Prize-to-weight ratio for the most binding constraint
    binding_eff = eff[most_binding, item_indices]  # (100,)
    
    # Combined scarcity premium score
    premium_score = binding_eff * scarcity_premium  # (100,)
    
    # Normalize all metrics to mean 1
    def normalize(arr):
        arr = np.asarray(arr, dtype=np.float64)
        mean_val = np.mean(arr)
        if mean_val > 0 and np.isfinite(mean_val):
            return arr / mean_val
        else:
            return np.ones_like(arr)
    
    agg_norm = normalize(agg_ratio)
    geo_norm = normalize(scarcity_geo)
    prem_norm = normalize(premium_score)
    
    # Combine metrics with weights
    # Aggregate gives baseline density
    # Scarcity-weighted geo emphasizes tight constraints
    # Premium highlights items with high value in their most limiting constraint
    w_agg = 0.35
    w_geo = 0.35
    w_prem = 0.30
    
    scores = w_agg * agg_norm + w_geo * geo_norm + w_prem * prem_norm
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores), scores, eps)
    scores = np.maximum(scores, eps)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
