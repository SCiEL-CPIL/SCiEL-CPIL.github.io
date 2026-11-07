"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute marginal utility heuristic for ACO knapsack.
    
    For each item, estimate its opportunity cost by looking at the scarcity
    of the resources it consumes. Items that consume scarce resources (where
    top-efficient items have high weight) get penalized, while items that
    offer high prize relative to current resource scarcity are rewarded.
    
    Approach:
    1. For each constraint, compute the efficiency ratio (prize/weight) for all items.
    2. Identify the top-k most efficient items per constraint (these represent
       the 'benchmark' efficiency for that resource).
    3. Compute a scarcity score for each constraint based on how tightly the
       top items pack the capacity.
    4. For each item, compute a marginal utility score that rewards high prize
       and penalizes consumption of scarce resources.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Ensure no division by zero
    weights_safe = np.where(weights > 1e-12, weights, 1e-12)
    
    # Efficiency ratio per constraint per item: prize / weight
    efficiency = prizes / weights_safe  # shape (n_constraints, n_items)
    
    # For each constraint, find the top-k most efficient items
    # k is chosen as roughly capacity / average_weight to estimate how many
    # top items would fill the capacity
    top_k = 25  # capacity per constraint
    
    # Compute scarcity for each constraint:
    # A constraint is "scarce" if the sum of weights of top-k efficient items
    # is close to or exceeds capacity (meaning even the best items can't all fit)
    # We use the average efficiency of top items as a proxy for resource value
    
    # For each constraint, get top-k item indices by efficiency
    # and compute their total weight (how much of the capacity they'd consume)
    scarcity_scores = np.zeros(n_constraints)
    
    for c in range(n_constraints):
        eff_c = efficiency[c]
        # Get top-k items by efficiency
        top_indices = np.argpartition(eff_c, -top_k)[-top_k:]
        top_weights = weights[c, top_indices]
        top_prizes = prizes[c, top_indices]
        
        # Total weight of top-k items
        total_top_weight = np.sum(top_weights)
        total_top_prize = np.sum(top_prizes)
        
        # Scarcity: if top items' total weight is close to capacity, the resource
        # is scarce. We normalize by capacity (25).
        # Higher scarcity means the resource is more tightly constrained by top items.
        if total_top_weight > 0:
            scarcity_scores[c] = total_top_weight / 25.0
        else:
            scarcity_scores[c] = 1.0
    
    # Now compute marginal utility for each item:
    # For item j, its marginal cost is the weighted sum of scarcity of each
    # constraint it consumes, weighted by how much of each resource it uses.
    # Its marginal benefit is the total prize.
    
    # Resource consumption normalized by capacity
    # For each item and constraint, how much of the capacity does it consume?
    consumption = weights / 25.0  # shape (n_constraints, n_items), each in [0, 1] roughly
    
    # Weighted scarcity cost per item: sum over constraints of scarcity[c] * consumption[c, j]
    # This estimates the "opportunity cost" of using this item
    scarcity_weighted = np.sum(scarcity_scores[:, None] * consumption, axis=0)  # shape (n_items,)
    
    # Total prize per item
    total_prize = np.sum(prizes, axis=0)  # shape (n_items,)
    
    # Marginal utility: reward high prize, penalize high scarcity cost
    # We want items that give high prize per unit of scarce resource consumed
    # Use a ratio-like form but stabilized
    
    # Combine: prize / (scarcity_cost + epsilon)
    # But also incorporate the efficiency relative to the top items
    # Items that are close to top efficiency in the constraints they consume get bonus
    
    # Compute for each item its best efficiency across constraints it significantly consumes
    # Simplified: use the minimum efficiency across constraints (bottleneck)
    min_efficiency = np.min(efficiency, axis=0)  # shape (n_items,)
    
    # Heuristic: balance between total prize, scarcity-adjusted cost, and efficiency
    # Higher is more attractive
    
    # Primary signal: prize / scarcity_cost
    base_score = total_prize / (scarcity_weighted + 1e-9)
    
    # Secondary signal: efficiency (items that are efficient in all dimensions are good)
    eff_score = min_efficiency
    
    # Combine: use geometric mean or weighted sum
    # Normalize both to [0, 1] range for stable combination
    base_min = np.min(base_score)
    base_max = np.max(base_score)
    base_norm = (base_score - base_min) / (base_max - base_min + 1e-9)
    
    eff_min = np.min(eff_score)
    eff_max = np.max(eff_score)
    eff_norm = (eff_score - eff_min) / (eff_max - eff_min + 1e-9)
    
    # Weighted combination: prioritize scarcity-adjusted prize, with efficiency as tiebreaker
    heuristic = 0.7 * base_norm + 0.3 * eff_norm
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
