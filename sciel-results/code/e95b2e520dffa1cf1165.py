import numpy as np
import time
import heapq

def _greedy_fill(capacity, weights, values, exclude_idx=None):
    """Greedy fill by density, excluding a specific index. Returns (total_value, list_of_indices)."""
    n = len(weights)
    if capacity <= 0:
        return 0.0, []
    
    # Build list of (density, value, weight, index) for feasible items
    candidates = []
    for i in range(n):
        w = weights[i]
        v = values[i]
        if exclude_idx is not None and i == exclude_idx:
            continue
        if w > 0 and w <= capacity:
            candidates.append((v / w, v, w, i))
        elif w == 0 and v > 0:
            # Free value, always take
            candidates.append((float('inf'), v, w, i))
    
    if not candidates:
        return 0.0, []
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    total_val = 0.0
    selected = []
    rem_cap = capacity
    
    for density, v, w, idx in candidates:
        if w <= rem_cap:
            total_val += v
            selected.append(idx)
            rem_cap -= w
            if rem_cap <= 0:
                break
    
    return total_val, selected


def _solve_knapsack_bnb(capacity, weights, values, max_time=5.0):
    """
    Branch and bound for 0/1 knapsack.
    Returns (best_value, best_indices).
    Uses fractional knapsack as upper bound.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    # Filter to feasible items
    feasible_indices = []
    for i in range(n):
        if weights[i] > 0 and weights[i] <= capacity:
            feasible_indices.append(i)
        elif weights[i] == 0 and values[i] > 0:
            feasible_indices.append(i)
    
    if not feasible_indices:
        return 0.0, []
    
    # Sort by density descending
    items = []
    for i in feasible_indices:
        if weights[i] > 0:
            items.append((values[i] / weights[i], values[i], weights[i], i))
        else:
            items.append((float('inf'), values[i], weights[i], i))
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute fractional upper bound helper
    def fractional_bound(start_idx, cap):
        """Upper bound using fractional knapsack from start_idx onwards."""
        bound = 0.0
        rem_cap = cap
        for j in range(start_idx, len(items)):
            density, v, w, idx = items[j]
            if w <= rem_cap:
                bound += v
                rem_cap -= w
            else:
                if w > 0:
                    bound += v * (rem_cap / w)
                break
        return bound
    
    # B&B using max-heap (negative for min-heap behavior, but we want max)
    # State: (negative_bound, current_value, capacity_left, next_item_idx, selected_indices)
    # We use a heap where we pop the highest bound first
    
    best_value = 0.0
    best_indices = []
    
    # Initial greedy solution
    greedy_val, greedy_idx = _greedy_fill(capacity, weights, values)
    best_value = greedy_val
    best_indices = list(greedy_idx)
    
    start_time = time.time()
    
    # Heap elements: (-bound, current_value, cap_left, next_idx, tuple(selected))
    # We want to explore nodes with highest bound first
    heap = []
    initial_bound = fractional_bound(0, capacity)
    heapq.heappush(heap, (-initial_bound, 0.0, capacity, 0, ()))
    
    nodes_explored = 0
    max_nodes = 50000
    
    while heap and nodes_explored < max_nodes:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        neg_bound, cur_val, cap_left, next_idx, selected = heapq.heappop(heap)
        bound = -neg_bound
        nodes_explored += 1
        
        if bound <= best_value:
            continue
        
        if next_idx >= len(items):
            if cur_val > best_value:
                best_value = cur_val
                best_indices = list(selected)
            continue
        
        density, v, w, idx = items[next_idx]
        
        # Option 1: Include item
        if w <= cap_left:
            new_val = cur_val + v
            new_cap = cap_left - w
            new_selected = selected + (idx,)
            new_bound = new_val + fractional_bound(next_idx + 1, new_cap)
            if new_bound > best_value:
                heapq.heappush(heap, (-new_bound, new_val, new_cap, next_idx + 1, new_selected))
        
        # Option 2: Exclude item
        new_bound = cur_val + fractional_bound(next_idx + 1, cap_left)
        if new_bound > best_value:
            heapq.heappush(heap, (-new_bound, cur_val, cap_left, next_idx + 1, selected))
    
    return best_value, best_indices


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 10.0
    
    # Filter to feasible items
    feasible_indices = []
    for i in range(n):
        w = remaining_weights[i]
        v = remaining_values[i]
        if w > 0 and w <= remaining_capacity:
            feasible_indices.append(i)
        elif w == 0 and v > 0:
            return i  # Free value, take it immediately
    
    if not feasible_indices:
        return -1
    
    # Get arrays for feasible items only
    f_weights = np.array([remaining_weights[i] for i in feasible_indices], dtype=np.float64)
    f_values = np.array([remaining_values[i] for i in feasible_indices], dtype=np.float64)
    f_indices = np.array(feasible_indices, dtype=np.int64)
    
    m = len(f_indices)
    
    # Greedy baseline
    greedy_val, greedy_sel = _greedy_fill(remaining_capacity, f_weights, f_values)
    best_total = greedy_val
    best_first = f_indices[greedy_sel[0]] if greedy_sel else f_indices[0]
    
    # Try B&B on the full problem to get optimal solution
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        bnb_val, bnb_sel = _solve_knapsack_bnb(remaining_capacity, f_weights, f_values, 
                                                 max_time=min(30.0, DEADLINE - SAFETY_MARGIN - elapsed))
        if bnb_val > best_total:
            best_total = bnb_val
            if bnb_sel:
                # Pick the first item in B&B solution (by density order among selected)
                sel_set = set(bnb_sel)
                # Sort selected by density
                sel_sorted = sorted(bnb_sel, key=lambda i: f_values[i] / f_weights[i] if f_weights[i] > 0 else float('inf'), reverse=True)
                best_first = f_indices[sel_sorted[0]]
    
    # If B&B didn't help or took too long, try candidate-based approach
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN and best_total <= greedy_val + 1e-9:
        # Try each top candidate as first pick with greedy completion
        # Sort by density
        densities = f_values / f_weights
        top_candidates = np.argsort(-densities)[:min(30, m)]
        
        for cand_local in top_candidates:
            if remaining_weights[f_indices[cand_local]] > remaining_capacity:
                continue
            
            rem_cap = remaining_capacity - f_weights[cand_local]
            # Greedy fill remaining
            val, _ = _greedy_fill(rem_cap, f_weights, f_values, exclude_idx=cand_local)
            total = f_values[cand_local] + val
            
            if total > best_total:
                best_total = total
                best_first = f_indices[cand_local]
    
    # Local search polish if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Reconstruct current best solution
        if best_total > greedy_val + 1e-9:
            # We have a B&B solution, try to improve with local search
            # Get the solution set
            # Re-run B&B with tighter time to get the solution
            pass
        else:
            # Use greedy solution and try swaps
            greedy_val2, greedy_sel2 = _greedy_fill(remaining_capacity, f_weights, f_values)
            current_sel = set(greedy_sel2)
            current_val = greedy_val2
            
            improved = True
            max_iters = 50
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                elapsed = time.time() - start_time
                if elapsed > DEADLINE - SAFETY_MARGIN:
                    break
                
                # Try 1-swap: remove one, add one
                for out_local in list(current_sel):
                    w_out = f_weights[out_local]
                    v_out = f_values[out_local]
                    new_cap = remaining_capacity - sum(f_weights[i] for i in current_sel if i != out_local)
                    
                    for in_local in range(m):
                        if in_local in current_sel:
                            continue
                        if f_weights[in_local] <= new_cap:
                            new_val = current_val - v_out + f_values[in_local]
                            if new_val > current_val + 1e-9:
                                current_sel.remove(out_local)
                                current_sel.add(in_local)
                                current_val = new_val
                                improved = True
                                break
                    if improved:
                        break
            
            if current_val > best_total:
                best_total = current_val
                # Pick highest density item from solution
                sel_sorted = sorted(current_sel, key=lambda i: f_values[i] / f_weights[i] if f_weights[i] > 0 else float('inf'), reverse=True)
                if sel_sorted:
                    best_first = f_indices[sel_sorted[0]]
    
    return int(best_first)
