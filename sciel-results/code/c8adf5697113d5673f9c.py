import numpy as np
import time

_START = time.monotonic()
_TIME_LIMIT = 110.0 # Safety margin

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Check time
    if time.monotonic() - _START > _TIME_LIMIT:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    # Feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # Calculate density for all items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    densities[weights == 0] = np.inf
    
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    # Precompute prefix sums for fast greedy completion estimation
    # sorted_weights[k] = weights[sorted_indices[k]]
    # sorted_values[k] = values[sorted_indices[k]]
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    
    # Prefix sums: prefix_w[k] = sum(sorted_weights[:k])
    # We want to find the largest k such that sum(sorted_weights[:k]) <= cap
    # This gives the value of the greedy fill assuming we can take the first k items.
    # Note: This is an approximation because in the actual greedy fill, we skip items
    # that don't fit individually, but since we're filling a remaining capacity with
    # items sorted by density, and we know the total weight of the first k items fits,
    # it's a reasonable upper bound or close estimate.
    # Actually, the greedy algorithm takes items in order if they fit.
    # If we just sum the first k items, that's equivalent to the greedy result IF
    # all items in the first k fit individually. Since they are sorted by density,
    # not weight, a heavy item might appear early.
    # However, for a heuristic score, using the prefix sum of the first k items
    # where total weight <= cap is a very fast and decent proxy for the greedy value.
    
    prefix_w = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    prefix_v = np.concatenate(([0.0], np.cumsum(sorted_values)))
    
    # For each candidate i, we need to estimate the value of filling (remaining_capacity - weights[i])
    # with items from sorted_indices, excluding i.
    # Excluding i is tricky for the prefix sum approach.
    # If i is not among the first k items, the prefix sum is valid.
    # If i is among the first k items, we need to subtract its value and weight,
    # and potentially adjust k.
    
    # Let's try a simpler approach: 
    # For each candidate i, calculate the greedy fill value exactly but faster?
    # The previous loop was O(N^2). With N=100, that's 10,000 ops per call.
    # If called ~50 times, that's 500,000 ops. This is fast enough in Python?
    # 500k simple ops is < 0.1s. So the O(N^2) loop might be fine.
    # The current code IS the O(N^2) loop.
    
    # Why was the score 1.0015? Maybe it's already good.
    # Let's look at the "INSPIRATIONS".
    # Inspiration 3: "lookahead greedy approach that estimates the total achievable value... known improvement over pure density greedy"
    # The current code DOES this: it simulates selection + density-greedy fill.
    
    # Can we improve the "greedy fill" part?
    # The current fill is: iterate through sorted_indices, if weight[j] <= rem_cap, take it.
    # This is the standard density greedy.
    
    # Alternative: What if we use a better completion heuristic?
    # E.g., after taking the first item, what if we try to fill with the *best* remaining items
    # not just by density, but by some other metric?
    # Or, what if we allow the "completion" to be a 0/1 knapsack solve on the remaining items?
    # No, that's too slow for N=100 if done repeatedly.
    
    # Let's stick to the current design but optimize the inner loop or the evaluation.
    # One potential improvement: The inner loop iterates over ALL sorted_indices.
    # We can break early if rem_cap becomes 0.
    
    # Also, we can pre-filter sorted_indices to only include items that are feasible
    # for the *original* remaining capacity? No, because rem_cap changes.
    # But items with weight > remaining_capacity can never be picked in the completion
    # for ANY candidate i (since rem_cap <= remaining_capacity).
    # So we can filter sorted_indices to only include items with weight <= remaining_capacity.
    
    # Let's implement this optimization.
    
    # Filter sorted_indices to only those that could possibly fit in the max remaining capacity
    # (which is remaining_capacity).
    # Actually, for candidate i, rem_cap = remaining_capacity - weights[i].
    # So items with weight > remaining_capacity can never be picked.
    # We can create a list of "usable" sorted indices.
    
    usable_mask = sorted_weights <= remaining_capacity
    usable_sorted_indices = sorted_indices[usable_mask]
    usable_sorted_weights = sorted_weights[usable_mask]
    usable_sorted_values = sorted_values[usable_mask]
    
    best_value = -1
    best_idx = -1
    
    for i in feasible_indices:
        cur_val = values[i]
        rem_cap = remaining_capacity - weights[i]
        
        # Greedily fill rem_cap
        for k in range(len(usable_sorted_indices)):
            j = usable_sorted_indices[k]
            if j == i:
                continue
            if usable_sorted_weights[k] <= rem_cap:
                cur_val += usable_sorted_values[k]
                rem_cap -= usable_sorted_weights[k]
                if rem_cap <= 0:
                    break
            
        if cur_val > best_value:
            best_value = cur_val
            best_idx = i
            
    return int(best_idx)
