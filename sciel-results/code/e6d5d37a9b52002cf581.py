"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a marginal utility / scarcity-adjusted heuristic.
    
    For each constraint c, we estimate the 'scarcity' or 'shadow price' by looking at
    the efficiency of the best items. Items that consume scarce resources heavily are
    penalized more.
    
    Approach:
    1. Compute efficiency ratio for each item per constraint: prize[c,i] / weight[c,i]
    2. For each constraint c, compute a 'scarcity score' based on the distribution of efficiencies.
       Specifically, we use the mean efficiency of the top 20% of items for that constraint
       as a proxy for the 'value per unit' of that resource.
    3. The 'cost' of an item is the sum over constraints of: weight[c,i] * scarcity_score[c]
    4. The 'value' of an item is the sum of prizes (or a weighted sum based on scarcity).
    5. Heuristic = value / (cost + epsilon)
    
    This rewards items that have high total prize relative to the 'true cost' of the
    resources they consume, where the cost is weighted by how scarce those resources are.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Avoid division by zero
    weights_safe = np.where(weights > 1e-9, weights, 1e-9)
    
    # Efficiency per constraint per item: prize / weight
    efficiency = prizes / weights_safe  # shape (n_constraints, n_items)
    
    # For each constraint, estimate scarcity.
    # We use the mean of the top 20% efficiencies as a proxy for the 'marginal value'
    # of that resource. Resources where the best items have very high efficiency are
    # 'scarce' in the sense that using them on a low-efficiency item is costly.
    # Actually, if the top items have high efficiency, the resource is 'valuable' to use
    # on those items, so using it on others is a bigger opportunity cost.
    
    # Let's compute a scarcity weight for each constraint.
    # We'll use the mean of the top 20% of efficiencies.
    top_k = max(1, n_items // 5)
    scarcity_scores = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_effs = np.sort(efficiency[c, :])[-top_k:]
        scarcity_scores[c] = np.mean(top_effs)
    
    # Normalize scarcity scores so they are comparable
    # If a constraint has high scarcity, its weight should be amplified in the cost calculation
    # Let's use the scarcity scores directly as multipliers for the weights in the cost function.
    
    # Cost of item i = sum_c (weight[c, i] * scarcity_scores[c])
    # Value of item i = sum_c (prize[c, i])
    
    # However, we might also want to adjust the value based on scarcity.
    # If a resource is scarce, prize from that resource is more valuable?
    # Or is it that the cost is higher?
    # The opportunity cost interpretation: if I use weight w of resource c, I lose the
    # ability to use that weight on the 'best' items for c, which would have given
    # w * scarcity_scores[c] prize.
    # So the 'net gain' of item i is:
    # sum_c (prize[c, i] - weight[c, i] * scarcity_scores[c])
    # But this can be negative. We want positive heuristics.
    # So maybe: Heuristic_i = max(0, sum_c (prize[c, i] - weight[c, i] * scarcity_scores[c])) + epsilon
    # Or: Heuristic_i = (sum_c prize[c, i]) / (sum_c weight[c, i] * scarcity_scores[c] + epsilon)
    
    # Let's try the ratio approach first, as it's more robust to scale.
    cost = np.sum(weights_safe * scarcity_scores[:, np.newaxis], axis=0)  # shape (n_items,)
    value = np.sum(prizes, axis=0)  # shape (n_items,)
    
    heuristic = value / (cost + 1e-9)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
