import numpy as np
import time
import heapq

def _solve_knapsack_bnb(capacity, weights, values):
    """
    Branch and bound for 0/1 knapsack.
    Returns (optimal_value, list_of_indices_included).
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    # Sort by value density descending
    idx_sorted = np.argsort(-values / weights)
    sorted_w = weights[idx_sorted].astype(np.float64)
    sorted_v = values[idx_sorted].astype(np.float64)
    
    # Precompute suffix sums for fractional upper bound
    suffix_w = np.zeros(n + 1)
    suffix_v = np.zeros(n + 1)
    suffix_w[n-1] = sorted_w[n-1]
    suffix_v[n-1] = sorted_v[n-1]
    for i in range(n-2, -1, -1):
        suffix_w[i] = suffix_w[i+1] + sorted_w[i]
        suffix_v[i] = suffix_v[i+1] + sorted_v[i]
    
    # Fractional bound: given starting index and remaining capacity,
    # take items greedily by density (already sorted)
    def fractional_bound(start_idx, rem_cap):
        if start_idx >= n or rem_cap <= 0:
            return 0.0
        bound = 0.0
        for i in range(start_idx, n):
            if sorted_w[i] <= rem_cap:
                bound += sorted_v[i]
                rem_cap -= sorted_w[i]
            else:
                bound += sorted_v[i] * (rem_cap / sorted_w[i])
                break
        return bound
    
    # Initial greedy solution
    greedy_val = 0.0
    greedy_indices = []
    cap = capacity
    for i in range(n):
        if sorted_w[i] <= cap:
            greedy_val += sorted_v[i]
            cap -= sorted_w[i]
            greedy_indices.append(i)
    
    # B&B with max-heap (use negative for min-heap)
    # Each entry: (-bound, start_idx, current_value, current_weight, included_set)
    # We use a frozenset or list for included items (indices in sorted order)
    
    best_value = greedy_val
    best_items = list(greedy_indices)
    
    # Start with root node
    heap = []
    root_bound = fractional_bound(0, capacity)
    heapq.heappush(heap, (-root_bound, 0, 0.0, 0.0, frozenset()))
    
    max_nodes = 5000  # Safety limit
    
    while heap and max_nodes > 0:
        max_nodes -= 1
        neg_bound, idx, val, wt, included = heapq.heappop(heap)
        bound = -neg_bound
        
        if bound <= best_value:
            continue
        
        if idx >= n:
            if val > best_value:
                best_value = val
                best_items = list(included)
            continue
        
        # Branch 1: Include item idx
        if sorted_w[idx] <= capacity - wt:
            new_wt = wt + sorted_w[idx]
            new_val = val + sorted_v[idx]
            new_included = included | {idx}
            next_idx = idx + 1
            if next_idx >= n:
                if new_val > best_value:
                    best_value = new_val
                    best_items = list(new_included)
            else:
                bound_next = new_val + fractional_bound(next_idx, capacity - new_wt)
                if bound_next > best_value:
                    heapq.heappush(heap, (-bound_next, next_idx, new_val, new_wt, new_included))
        
        # Branch 2: Exclude item idx
        next_idx = idx + 1
        if next_idx >= n:
            if val > best_value:
                best_value = val
                best_items = list(included)
        else:
            bound_next = val + fractional_bound(next_idx, capacity - wt)
            if bound_next > best_value:
                heapq.heappush(heap, (-bound_next, next_idx, val, wt, included))
    
    # Map back to original indices
    result_indices = [int(idx_sorted[i]) for i in best_items]
    return best_value, result_indices


def _greedy_density(capacity, weights, values):
    """Simple density greedy."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    idx_sorted = np.argsort(-values / weights)
    total = 0.0
    cap = capacity
    result = []
    for i in idx_sorted:
        if weights[i] <= cap:
            total += values[i]
            cap -= weights[i]
            result.append(int(i))
    return total, result


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_idx = np.where(feasible_mask)[0]
    
    if len(feasible_idx) == 0:
        return -1
    
    # Try B&B with time limit
    start_time = time.time()
    time_limit = 12.0  # 12 seconds per call, with 10 calls that's 120s
    
    try:
        opt_val, opt_indices = _solve_knapsack_bnb(remaining_capacity, remaining_weights, remaining_values)
    except Exception:
        opt_val, opt_indices = _greedy_density(remaining_capacity, remaining_weights, remaining_values)
    
    elapsed = time.time() - start_time
    
    if len(opt_indices) == 0:
        # Fall back to density greedy
        _, greedy_indices = _greedy_density(remaining_capacity, remaining_weights, remaining_values)
        if len(greedy_indices) == 0:
            return -1
        return greedy_indices[0]
    
    # Pick the item from the optimal solution that has highest density
    # to make consistent choices
    best_density = -1
    best_choice = opt_indices[0]
    for idx in opt_indices:
        dens = remaining_values[idx] / remaining_weights[idx]
        if dens > best_density:
            best_density = dens
            best_choice = idx
    
    return int(best_choice)
