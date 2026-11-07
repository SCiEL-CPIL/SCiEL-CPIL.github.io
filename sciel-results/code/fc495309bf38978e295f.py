"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    We want items that give high prize while consuming few resources in ALL constraints.
    
    Key insights:
    1. For each item, the "effective cost" is determined by the tightest constraint it hits.
       An item is limited by the constraint j where weights[j, i] / capacity_j is largest.
       Since all capacities are 25, this is just the max weight across constraints.
       
    2. Prize per unit of the binding resource: prize[i] / max_j(weights[j, i])
       This captures how much prize you get per unit of your most scarce resource.
       
    3. Additionally, items with lower total weight across all constraints are preferable
       because they leave more room for other items.
       
    4. We combine:
       - Primary: prize / (max weight across constraints)  -- bottleneck efficiency
       - Secondary: prize / (sum of weights across constraints) -- aggregate efficiency
       - Tertiary: a small penalty for items that are heavy in multiple constraints
         (diversity penalty)
       
    All components are normalized to mean 1 before combining.
    """
    eps = 1e-9
    
    # prizes shape: (5, 100), weights shape: (5, 100)
    # Each row is a constraint, each column is an item
    # prizes[j, i] = prize of item i under constraint j? 
    # Actually, re-reading: "prizes and weights, of shape (5, 100)"
    # The task says "prizes and weights, of shape (5, 100), are uniform in [0, 1]"
    # And "Each constraint has capacity 25"
    
    # Wait, let me re-read the problem more carefully.
    # "Item scores for ant colony optimization of the multiple-constraint knapsack"
    # "prizes and weights, of shape (5, 100)"
    # "Every constraint has capacity 25"
    
    # So we have 5 constraints and 100 items.
    # prizes[j, i] likely means the prize value of item i (but why 5 rows?)
    # Actually, in a multiple-constraint knapsack, each item has a single prize value,
    # but consumes weight in each of the 5 constraints.
    # So prizes might be (5, 100) but perhaps all rows are the same? Or maybe
    # prizes is the prize per constraint? That doesn't make much sense.
    
    # Let me re-read: "prizes and weights, of shape (5, 100), are uniform in [0, 1]"
    # "Each constraint has capacity 25"
    # "Each ant adds items that still fit in all constraints"
    # "best prize of that iteration"
    
    # I think prizes[j, i] might be the prize contribution of item i to constraint j,
    # but that's unusual. More likely, the prize is a single value per item, and
    # the shape (5, 100) for prizes might mean something else.
    
    # Actually, looking at the current program:
    # total_weights = weights.sum(axis=0)  # shape (100,)
    # sum_ratio = prizes / (total_weights + eps)  # this would be (5,100)/(100,) = (5,100)
    # Then they take .mean() which would give (5,) not (100,)
    
    # Wait, that doesn't match. Let me look again.
    # sum_ratio = prizes / (total_weights + eps)
    # If prizes is (5, 100) and total_weights is (100,), broadcasting gives (5, 100)
    # Then sum_ratio.mean() would be a scalar, not (100,)
    # But then sum_ratio_norm = sum_ratio / sum_ratio_mean would be (5, 100)
    # And scores would be (5, 100), not (100,)
    
    # That can't be right. The function should return "one score per item" i.e. (100,)
    
    # Let me re-examine. Maybe prizes is actually (100,) and weights is (5, 100)?
    # But the task says both are (5, 100).
    
    # Hmm, let me just assume that prizes[i] is the prize of item i, and it's given
    # as a (5, 100) array where all rows are identical (or we should average them).
    # Or maybe prizes[j, i] is the prize of item i, and there are 5 different prize
    # vectors? That seems odd.
    
    # Actually, I think the most sensible interpretation is:
    # - prizes is (5, 100) but represents the same prize for each item across constraints
    #   (i.e., prizes[0] = prizes[1] = ... = prizes[4] = prize_vector of length 100)
    # - weights is (5, 100) where weights[j, i] is the weight of item i in constraint j
    
    # Let me just use the mean across the first axis for prizes to get a (100,) vector.
    
    prize_per_item = prizes.mean(axis=0)  # (100,)
    
    # weights[j, i] = weight of item i in constraint j
    # max_weight[i] = max_j weights[j, i]  -- the binding constraint weight
    max_weight = weights.max(axis=0)  # (100,)
    sum_weight = weights.sum(axis=0)  # (100,)
    
    # 1. Bottleneck efficiency: prize / max_weight
    bottleneck_eff = prize_per_item / (max_weight + eps)  # (100,)
    
    # 2. Aggregate efficiency: prize / sum_weight
    aggregate_eff = prize_per_item / (sum_weight + eps)  # (100,)
    
    # 3. Count how many constraints are "tight" for each item (weight > 0.5 * max)
    # This penalizes items that consume resources in many constraints
    tight_count = (weights > 0.5 * max_weight[np.newaxis, :]).sum(axis=0).astype(float)  # (100,)
    # Normalize tight_count to [0, 1] range roughly
    tight_penalty = 1.0 / (1.0 + tight_count)  # (100,)
    
    # Normalize each component to mean 1
    def normalize_mean_one(x):
        m = x.mean()
        if m > eps:
            return x / m
        else:
            return np.ones_like(x)
    
    bottleneck_norm = normalize_mean_one(bottleneck_eff)
    aggregate_norm = normalize_mean_one(aggregate_eff)
    tight_norm = normalize_mean_one(tight_penalty)
    
    # Combine with weights
    # Bottleneck efficiency is most important (determines if item can fit)
    # Aggregate efficiency is secondary
    # Tightness penalty is a small adjustment
    scores = 0.6 * bottleneck_norm + 0.3 * aggregate_norm + 0.1 * tight_norm
    
    # Ensure positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
