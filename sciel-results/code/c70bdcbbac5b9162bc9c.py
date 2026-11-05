"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining arrays, or -1 to stop.
    
    Strategy:
    1. If no items remain, return -1.
    2. Compute value density (v/w) for each item.
    3. Use a composite score that balances density, absolute value, and fit.
       - High density is generally good.
       - High absolute value is good if it fits.
       - We penalize items that are too heavy relative to remaining capacity
         unless they have extremely high value.
    4. Specifically, we use a score that approximates the marginal gain.
       A common strong heuristic is "density" but with a tie-breaker or
       adjustment for "waste".
       
       Let's try a refined density approach:
       score = (value / weight) * (1 + value / max_value)
       This boosts high-value items slightly more than pure density.
       
       Alternatively, we can look at the "efficiency" of filling the remaining capacity.
       If an item fits exactly or very well, it's good.
       
       Let's stick to a robust variant of density with a small boost for high value.
       Actually, the seed program (pure density) scored 1.0. 
       Often, "value" alone or "density" alone are standard.
       
       Let's try a hybrid:
       1. Filter items that fit (w <= remaining_capacity). Note: the input arrays 
          are described as "remaining (feasible) arrays" in the prompt? 
          "index into the remaining (feasible) arrays". 
          Wait, the prompt says: "index into the remaining (feasible) arrays".
          This implies that the caller might already filter out infeasible items?
          Or does it mean the arrays contain all remaining items, and we must pick a feasible one?
          
          Let's re-read carefully: 
          "select_next_item(remaining_capacity, remaining_weights, remaining_values) -> index into the remaining (feasible) arrays"
          
          Usually in these tasks, the inputs are the *current* state. If an item doesn't fit, 
          picking it would be invalid or lead to a dead end. 
          However, the function returns an index. If I return an index of an infeasible item, 
          the evaluator might crash or score 0. 
          
          Let's assume the inputs are the *entire* remaining set, and I must ensure the 
          returned index corresponds to an item with weight <= remaining_capacity.
          
          If no item fits, return -1.
          
          Heuristic:
          1. Identify feasible items: w[i] <= remaining_capacity.
          2. If none, return -1.
          3. Among feasible items, pick the one with the best score.
          
          Score options:
          a) Pure density: v/w
          b) Pure value: v
          c) Density * Value
          d) (v + epsilon) / (w + epsilon)
          
          Let's test "Density" vs "Value". 
          Pure density is the standard greedy for fractional, but for 0/1 it can be suboptimal.
          However, it's the seed. 
          
          A common improvement is to consider the "waste" of capacity.
          Let's try: Score = (v / w) * (remaining_capacity - w) ? No, that's weird.
          
          Let's try a simple robust heuristic:
          Sort by density descending.
          But we can also use a look-ahead or a composite score.
          
          Given the small N (100 items) and capacity (12.5), the items are likely small.
          
          Let's try: 
          1. Filter feasible.
          2. Calculate density.
          3. Calculate "fit" = 1 - (w / remaining_capacity) ? No.
          
          Let's try a score that favors high density but also high value.
          score = v / w + v * 0.01  # Slight boost to value
          
          Or, we can use the "ratio of value to weight" but break ties by value.
          
          Actually, a very strong heuristic for small capacities is to use the 
          "density" but if the top density item leaves a lot of waste, maybe a 
          slightly lower density item that fits better is better?
          
          Let's try this:
          1. Get feasible indices.
          2. Compute densities.
          3. Select the item with the maximum density.
          4. If there are ties or close scores, prefer higher value.
          
          Let's implement a slightly more sophisticated score:
          score = (v / w) * (1 + v / max_v)
          This amplifies the importance of high-value items.
          
          Let's also add a time guard, though it should be fast.
          """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Ensure inputs are numpy arrays for speed
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    cap = float(remaining_capacity)
    
    # Find feasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Calculate densities
    # Avoid division by zero (though weights should be > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = f_values / f_weights
    
    # Replace any inf/nan with 0 (or a large number if value is high and weight is 0, 
    # but weight 0 is usually not in knapsack problems. If w=0, v>0, it's free money, pick it.)
    # If w=0, density is inf. We should pick it immediately.
    if np.any(np.isinf(densities)):
        inf_idx = np.where(np.isinf(densities))[0]
        # Among infinite density (w=0), pick highest value
        best_inf = inf_idx[np.argmax(f_values[inf_idx])]
        return int(feasible_indices[best_inf])
    
    # Heuristic: Composite Score
    # We want to maximize value. 
    # Pure density is a good proxy.
    # Let's try to refine it.
    
    # Option 1: Pure Density
    # Option 2: Density * Value
    # Option 3: (Value / Weight) + (Value / MaxWeight)
    
    # Let's try a score that is primarily density, but breaks ties by value.
    # To make it a single scalar for argmax:
    # score = density * (1 + value / max_value)
    
    max_val = np.max(f_values)
    if max_val == 0:
        # All values 0? Pick any (or the lightest to save space? but value is 0 anyway)
        # Pick the lightest to leave room for future items? 
        # Since value is 0, it doesn't matter for score, but might matter for future.
        # However, if all values are 0, the total value is 0.
        # Let's pick the lightest one to maximize future feasibility.
        lightest_idx = np.argmin(f_weights)
        return int(feasible_indices[lightest_idx])
    
    # Normalize values to [0, 1]
    norm_values = f_values / max_val
    
    # Score: Density boosted by normalized value
    # This favors high density, but if two items have similar density, 
    # the one with higher absolute value wins.
    scores = densities * (1.0 + norm_values)
    
    # Pick the best score
    best_local_idx = int(np.argmax(scores))
    return int(feasible_indices[best_local_idx])

# EVOLVE-BLOCK-END
