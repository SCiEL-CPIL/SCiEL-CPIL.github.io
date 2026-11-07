"""Heuristics for ant colony optimization of the multiple-constraint knapsack."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Estimate shadow prices for each constraint based on resource scarcity,
    then score items by prize minus the weighted sum of resource costs (Lagrangian surplus).
    This penalizes items that consume scarce resources disproportionately.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,), one attractiveness score per item
    """
    n_constraints = prizes.shape[0]
    n_items = prizes.shape[1]
    capacity = 25.0
    
    # Total weight per constraint
    total_weights = weights.sum(axis=1)  # shape (5,)
    
    # Shadow price: inverse of total weight sum, scaled by capacity
    # Constraints with less total weight are more "scarce" per unit capacity
    # Actually, if total weight >> capacity, the constraint is tight
    # Shadow price should be higher for tighter constraints
    # Let's think: if total_weight is close to capacity, the constraint is binding
    # A simple proxy: shadow_price[c] = 1 / (total_weight[c] / capacity + epsilon)
    # This gives higher shadow price when total_weight is smaller relative to capacity
    # But actually, if total_weight >> capacity, the constraint is very tight
    # Let's use: shadow_price[c] = capacity / total_weight[c]
    # If total_weight = capacity, shadow_price = 1
    # If total_weight = 2*capacity, shadow_price = 0.5
    # If total_weight = 0.5*capacity, shadow_price = 2
    # This seems counterintuitive. Let me reconsider.
    
    # Actually, for a knapsack, if the total weight of all items is much larger than capacity,
    # the constraint is very binding, so each unit of weight is very expensive.
    # Shadow price should be high when the constraint is tight.
    # Let's use: shadow_price[c] = 1 / (total_weight[c] - capacity + epsilon) if total_weight > capacity
    # Or simpler: shadow_price[c] = capacity / total_weight[c]
    # When total_weight is large relative to capacity, shadow_price is small? That's wrong.
    
    # Let me think differently. The "scarcity" of a constraint is how much it limits the solution.
    # If total_weight[c] >> capacity, the constraint is very restrictive.
    # A good proxy for shadow price is: shadow_price[c] = 1 / (total_weight[c] / capacity)
    # = capacity / total_weight[c]
    # This gives shadow_price = 1 when total_weight = capacity
    # shadow_price = 0.5 when total_weight = 2*capacity
    # shadow_price = 2 when total_weight = 0.5*capacity
    # Hmm, this still seems backwards.
    
    # Let's use a different approach: the fraction of capacity used if we took all items
    # utilization[c] = total_weight[c] / capacity
    # If utilization > 1, the constraint is binding
    # Shadow price should be higher for more binding constraints
    # Let's use: shadow_price[c] = max(1, utilization[c])
    # Or: shadow_price[c] = utilization[c] ** 2 to emphasize tight constraints
    
    # Actually, the simplest and most robust approach:
    # shadow_price[c] = 1 / (total_weight[c] + epsilon)
    # This gives higher shadow price for constraints with less total weight available
    # But that's also counterintuitive.
    
    # Let me step back. The key insight is:
    # - If a constraint has a high total weight relative to its capacity, it's tight
    # - Items that consume more of a tight constraint should be penalized more
    # - So shadow_price[c] should be higher when the constraint is tighter
    
    # Let's define tightness[c] = total_weight[c] / capacity
    # If tightness > 1, the constraint is binding
    # shadow_price[c] = tightness[c] (or some monotonic function of it)
    
    tightness = total_weights / capacity  # shape (5,)
    # Ensure shadow prices are positive
    shadow_prices = np.maximum(tightness, 1.0)  # shape (5,)
    
    # Compute Lagrangian surplus for each item
    # surplus[item] = prize[item] - sum_c(shadow_price[c] * weight[c, item])
    # prizes shape: (5, 100), weights shape: (5, 100)
    # For each item j: surplus[j] = mean_c(prizes[c, j]) - sum_c(shadow_price[c] * weights[c, j])
    
    # Use mean prize across constraints as the "value" of the item
    mean_prize = prizes.mean(axis=0)  # shape (100,)
    
    # Weighted resource cost: sum_c(shadow_price[c] * weights[c, j])
    # shadow_prices shape: (5,), weights shape: (5, 100)
    weighted_cost = (shadow_prices[:, np.newaxis] * weights).sum(axis=0)  # shape (100,)
    
    # Lagrangian surplus
    surplus = mean_prize - weighted_cost  # shape (100,)
    
    # Convert to positive scores: add a constant to shift to positive range
    # Then apply a power or exponential to sharpen the distribution
    min_surplus = surplus.min()
    shifted = surplus - min_surplus + 1e-6  # Ensure positive
    
    # Apply a power to sharpen the gradient (items with higher surplus get disproportionately higher scores)
    # Using a power > 1 amplifies differences
    scores = shifted ** 1.5
    
    # Ensure all scores are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
