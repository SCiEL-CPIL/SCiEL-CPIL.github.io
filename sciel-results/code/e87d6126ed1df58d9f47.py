import numpy as np
import time

_START = None
_HARD_LIMIT = 142.0  # Safety margin before 150s kill

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time()

def _time_left():
    return _HARD_LIMIT - (_now() - _START)

def _density_greedy_solution(cap, weights, values):
    """
    Returns a boolean mask of items selected by density greedy,
    and the total value.
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    
    cap = float(cap)
    selected = np.zeros(n, dtype=bool)
    total_val = 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    order = np.argsort(-density, kind='stable')
    
    for idx in order:
        idx = int(idx)
        wi = float(weights[idx])
        if wi <= cap:
            cap -= wi
            total_val += float(values[idx])
            selected[idx] = True
        if cap <= 1e-9:
            break
            
    return selected, total_val

def _local_search(cap, weights, values, selected):
    """
    Perform local search on the selected set to improve total value.
    Uses 1-for-1 swaps and 2-for-1 swaps (add 2, remove 1) if beneficial.
    Returns the improved boolean mask.
    """
    n = len(weights)
    if n == 0:
        return selected
        
    selected = selected.copy()
    improved = True
    iterations = 0
    
    while improved and _time_left() > 0.1:
        improved = False
        iterations += 1
        if iterations > 50: # Safety break for infinite loops in degenerate cases
            break
            
        # Calculate current stats
        total_weight = float(np.sum(weights[selected]))
        total_value = float(np.sum(values[selected]))
        
        # 1. 1-for-1 Swaps: Remove selected item i, add unselected item j
        # Gain = value[j] - value[i]
        # Constraint: weight[j] - weight[i] <= remaining_capacity (which is cap - total_weight)
        # Actually, since we are swapping, the new total weight is total_weight - w[i] + w[j] <= cap
        # So w[j] - w[i] <= cap - total_weight
        
        remaining_cap = cap - total_weight
        
        # Indices of selected and unselected
        sel_idx = np.where(selected)[0]
        unsel_idx = np.where(~selected)[0]
        
        if len(sel_idx) == 0 or len(unsel_idx) == 0:
            break
            
        best_gain = 1e-9 # Must be strictly better to avoid oscillation
        best_swap = None
        
        # Vectorized-ish check for 1-for-1
        # For each unselected j, we want to find a selected i such that:
        # values[j] - values[i] > best_gain
        # AND weights[j] - weights[i] <= remaining_cap
        
        # This is O(N^2) in worst case if done naively.
        # With N=100, N^2 = 10,000. This is fast.
        # Let's just loop.
        
        for j in unsel_idx:
            wj = float(weights[j])
            vj = float(values[j])
            
            # We need wj - wi <= remaining_cap  => wi >= wj - remaining_cap
            min_wi_needed = wj - remaining_cap
            
            # Find all selected i where weights[i] >= min_wi_needed
            # And maximize vj - vi  => minimize vi
            # So we want the selected item i with minimum value among those satisfying weight constraint.
            
            # Get weights and values of selected items
            sel_w = weights[sel_idx]
            sel_v = values[sel_idx]
            
            # Mask: sel_w >= min_wi_needed
            mask = sel_w >= min_wi_needed
            if not np.any(mask):
                continue
                
            candidate_v = sel_v[mask]
            min_v_idx_in_mask = np.argmin(candidate_v)
            min_v = candidate_v[min_v_idx_in_mask]
            
            gain = vj - min_v
            if gain > best_gain:
                # Find the original index of this selected item
                # sel_idx is the array of indices. 
                # mask is applied to sel_w/sel_v.
                # We need the index in sel_idx corresponding to min_v_idx_in_mask.
                # But wait, mask filters sel_idx.
                # Let's be precise.
                
                # Let's just iterate to be safe and simple, N=100 is small.
                # Actually, let's stick to the vectorized logic but get the index right.
                # The indices of the masked selected items are:
                masked_sel_indices = sel_idx[mask]
                target_i = masked_sel_indices[min_v_idx_in_mask]
                
                best_gain = gain
                best_swap = (int(target_i), int(j))
        
        if best_swap is not None:
            i, j = best_swap
            selected[i] = False
            selected[j] = True
            improved = True
            # Update sel_idx/unsel_idx for next iteration? 
            # The loop recalculates them at the start of the while loop.
            continue # Restart search with new state
            
        # 2. 2-for-1 Swaps: Remove selected i, add unselected j, k
        # Gain = v[j] + v[k] - v[i]
        # Constraint: w[j] + w[k] - w[i] <= remaining_cap
        # This is O(N^3) worst case (100^3 = 1M). 
        # 1M operations in Python loop might be ~0.1s. 
        # We have ~140s. 
        # However, we do this inside a while loop. 
        # If we do 100 iterations of local search, that's 100M ops. Might be too slow in pure Python.
        # Let's optimize or limit 2-for-1.
        
        # Given N=100, let's try a limited 2-for-1 search.
        # Only try if we have time and 1-for-1 didn't improve.
        
        if _time_left() > 1.0 and len(unsel_idx) >= 2:
            # We want to find i, j, k such that:
            # v[j] + v[k] - v[i] is max
            # w[j] + w[k] - w[i] <= remaining_cap
            
            # Precompute pair sums for unselected items?
            # There are ~50 unselected items. Pairs ~ 1250.
            # For each pair (j, k), we need to find a selected i such that:
            # w[i] >= w[j] + w[k] - remaining_cap
            # And we want to minimize v[i] to maximize gain.
            
            best_gain_21 = 1e-9
            best_swap_21 = None
            
            # Iterate over pairs of unselected items
            u_list = list(unsel_idx)
            L = len(u_list)
            
            # To keep it fast, we might just sample or limit.
            # But 1250 pairs * 100 selected checks = 125k ops. Very fast.
            
            for a in range(L):
                j = u_list[a]
                wj = float(weights[j])
                vj = float(values[j])
                
                for b in range(a + 1, L):
                    k = u_list[b]
                    wk = float(weights[k])
                    vk = float(values[k])
                    
                    w_pair = wj + wk
                    v_pair = vj + vk
                    
                    # Need w[i] >= w_pair - remaining_cap
                    min_wi_needed = w_pair - remaining_cap
                    
                    sel_w = weights[sel_idx]
                    sel_v = values[sel_idx]
                    
                    mask = sel_w >= min_wi_needed
                    if not np.any(mask):
                        continue
                        
                    candidate_v = sel_v[mask]
                    min_v = np.min(candidate_v)
                    
                    gain = v_pair - min_v
                    if gain > best_gain_21:
                        masked_sel_indices = sel_idx[mask]
                        target_i = masked_sel_indices[np.argmin(candidate_v)]
                        best_gain_21 = gain
                        best_swap_21 = (int(target_i), int(j), int(k))
            
            if best_swap_21 is not None:
                i, j, k = best_swap_21
                selected[i] = False
                selected[j] = True
                selected[k] = True
                improved = True
                continue # Restart search
        
    return selected

def _fast_density_index(weights, values, feasible):
    """O(n) fallback: pick max value/weight among feasible."""
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 0, values / weights, 0.0)
    fd = np.where(feasible, density, -np.inf)
    return int(np.argmax(fd))

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity is None:
        return -1
    cap = float(remaining_capacity)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    if n == 0 or cap <= 0:
        return -1
    
    feasible = weights <= cap
    if not feasible.any():
        return -1
    
    # 1. Generate initial solution using density greedy
    selected, total_val = _density_greedy_solution(cap, weights, values)
    
    # 2. Improve using local search
    if _time_left() > 0.5:
        selected = _local_search(cap, weights, values, selected)
    
    # 3. Determine which item to return.
    # The contract says "picks the next item to add".
    # In a greedy construction, we add items one by one.
    # If we have a full set `selected`, which one is the "next" one?
    # Usually, the algorithm builds the set incrementally.
    # If we return the item with the highest index in `selected`, 
    # and the evaluator adds it to the current partial solution (which is empty at start, 
    # then grows), does that make sense?
    
    # Wait, the evaluator calls select_next_item repeatedly.
    # Call 1: remaining = all items. We return index i1.
    # Evaluator adds i1 to solution.
    # Call 2: remaining = all items except i1. We return index i2 (from remaining array).
    # ...
    
    # My local search operates on the FULL remaining set to find the BEST SUBSET.
    # If I return the "last" item of that subset, is it consistent?
    # If the evaluator is building a set, and I say "add item X", and then in the next step
    # I calculate the best subset of the REMAINING items (excluding X), 
    # I am effectively doing:
    # Step 1: Find best subset S1 of All. Return x1 in S1.
    # Step 2: Find best subset S2 of All\{x1}. Return x2 in S2.
    # This is NOT the same as returning the items of the optimal set S1 in order.
    # Because S2 is optimized for All\{x1}, not for S1\{x1}.
    # However, since S1 was the best of All, and we remove one item, S1\{x1} is a strong candidate for S2.
    # But S2 might be better.
    
    # The key is: which item to return?
    # If I return the item with the highest index in `selected`, 
    # and the evaluator removes it, the next call will re-optimize the rest.
    # This effectively constructs a solution by peeling off items from a locally optimal set.
    # Does the order matter for the final sum? No, the sum is the sum of values.
    # Does the order matter for feasibility? The evaluator maintains `remaining_capacity`.
    # If I pick an item that is in the optimal set, it is feasible.
    # As long as I always pick an item from the locally optimal set of the CURRENT remaining items,
    # I am guaranteed to pick a feasible item.
    
    # Which item to pick?
    # If I pick the "most critical" item (e.g., highest value), it might be better.
    # Or the "least critical" (lowest value) to leave the "good" ones for later optimization?
    # Actually, if I pick the item with the LOWEST value in the selected set, 
    # the remaining set will still contain the high-value items, which might form an even better set 
    # in the next step (since we are re-optimizing).
    # Conversely, if I pick the HIGHEST value item, the next step has to work around its absence.
    
    # Let's try picking the item with the lowest value in the selected set.
    # This seems like a "greedy on the greedy" approach.
    
    sel_indices = np.where(selected)[0]
    if len(sel_indices) == 0:
        # Fallback to density
        return _fast_density_index(weights, values, feasible)
        
    # Pick the index with the minimum value among selected
    # Tie-breaker: maximum weight (to free up space?) or just first.
    # Let's pick min value.
    min_val_idx_in_sel = np.argmin(values[sel_indices])
    target_idx = sel_indices[min_val_idx_in_sel]
    
    return int(target_idx)
