import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    start_time = time.time()
    deadline = start_time + 100.0  # 100s margin out of 150s total
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)
    
    if C <= 0:
        return -1

    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    n_f = len(w_f)
    
    if n_f == 0:
        return -1

    # Check for zero-weight items with positive value
    zero_w_mask = (w_f == 0) & (v_f > 0)
    if np.any(zero_w_mask):
        zero_indices = np.where(zero_w_mask)[0]
        return int(feasible_indices[zero_indices[0]])

    # Compute densities
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = 0.0
    
    # Sort by density descending
    sort_idx = np.argsort(-d_f)
    w_s = w_f[sort_idx]
    v_s = v_f[sort_idx]
    d_s = d_f[sort_idx]
    
    # Prefix sums for LP bound
    pw = np.cumsum(w_s)
    pv = np.cumsum(v_s)
    
    def get_lp_value(cap):
        if cap <= 1e-9:
            return 0.0
        k = np.searchsorted(pw, cap, side='right') - 1
        if k < 0:
            if len(w_s) == 0:
                return 0.0
            return cap * d_s[0]
        val = pv[k]
        remaining_cap = cap - pw[k]
        if k < len(w_s) - 1:
            next_w = w_s[k + 1]
            next_d = d_s[k + 1]
            val += remaining_cap * next_d
        return val

    # Strategy 1: Greedy by density
    # Strategy 2: Greedy by value
    # Strategy 3: Greedy by weight (lightest first)
    
    best_score = -1.0
    best_i_local = -1
    
    def simulate_greedy(capacity, weights, values, order):
        total_value = 0.0
        current_cap = capacity
        for idx in order:
            if weights[idx] > current_cap:
                continue
            total_value += values[idx]
            current_cap -= weights[idx]
            if current_cap <= 1e-9:
                break
        return total_value

    # Strategy 1: Density order
    density_order = sort_idx
    if time.time() < deadline:
        score1 = simulate_greedy(C, w_f, v_f, density_order)
        if score1 > best_score:
            best_score = score1
            best_i_local = density_order[0]
    
    # Strategy 2: Value order
    if time.time() < deadline:
        value_order = np.argsort(-v_f)
        score2 = simulate_greedy(C, w_f, v_f, value_order)
        if score2 > best_score:
            best_score = score2
            best_i_local = value_order[0]
    
    # Strategy 3: Weight order (lightest first)
    if time.time() < deadline:
        weight_order = np.argsort(w_f)
        score3 = simulate_greedy(C, w_f, v_f, weight_order)
        if score3 > best_score:
            best_score = score3
            best_i_local = weight_order[0]
            
    # Strategy 4: Branch and Bound
    if time.time() < deadline and n_f <= 200:
        # Use B&B on the sorted items
        # We'll search in the sorted order (by density)
        # State: (index, remaining_cap, current_value)
        # Use a priority queue ordered by upper bound (LP value from this state)
        
        import heapq
        
        # Upper bound for a state: current_value + LP_value(remaining_cap, items from index onwards)
        def upper_bound(idx, cap, cur_val):
            if idx >= n_f:
                return cur_val
            # LP relaxation on items idx..n_f-1 with capacity cap
            # Approximate: use density of item idx and remaining items
            # Better: compute LP on the subarray
            if cap <= 1e-9:
                return cur_val
            
            # Simple bound: cur_val + sum of values of items that fit in cap (greedy by density from idx)
            # This is an admissible bound (upper bound on achievable value)
            bound_val = cur_val
            rem_cap = cap
            for j in range(idx, n_f):
                if w_s[j] <= rem_cap:
                    bound_val += v_s[j]
                    rem_cap -= w_s[j]
                    if rem_cap <= 1e-9:
                        break
                else:
                    # Fractional part
                    bound_val += (rem_cap / w_s[j]) * v_s[j]
                    break
            return bound_val
        
        # B&B using max-heap on upper bound
        # Each entry: (-upper_bound, idx, cap, cur_val, chosen_set)
        # We want to find the best first item to pick
        
        # Initialize with root: idx=0, cap=C, cur_val=0
        heap = []
        root_ub = upper_bound(0, C, 0.0)
        heapq.heappush(heap, (-root_ub, 0, C, 0.0, []))
        
        best_solution_value = 0.0
        best_first_item = -1
        nodes_explored = 0
        max_nodes = 50000  # Limit to avoid timeout
        
        while heap and time.time() < deadline and nodes_explored < max_nodes:
            neg_ub, idx, cap, cur_val, chosen = heapq.heappop(heap)
            ub = -neg_ub
            nodes_explored += 1
            
            # Prune if upper bound doesn't exceed best
            if ub <= best_solution_value + 1e-9:
                continue
            
            # If we've processed all items, update best
            if idx >= n_f:
                if cur_val > best_solution_value:
                    best_solution_value = cur_val
                    if chosen:
                        best_first_item = chosen[0]
                continue
            
            # Option 1: Skip item idx
            skip_ub = upper_bound(idx + 1, cap, cur_val)
            if skip_ub > best_solution_value + 1e-9:
                heapq.heappush(heap, (-skip_ub, idx + 1, cap, cur_val, chosen))
            
            # Option 2: Take item idx (if it fits)
            if w_s[idx] <= cap + 1e-9:
                new_cap = cap - w_s[idx]
                new_val = cur_val + v_s[idx]
                new_chosen = chosen + [idx] if not chosen else chosen  # Keep only first for efficiency
                
                # If this is the first item, record it
                if not chosen and new_val > best_solution_value:
                    # This is a candidate first item
                    # But we need to see if we can improve further
                    pass
                
                take_ub = upper_bound(idx + 1, new_cap, new_val)
                if take_ub > best_solution_value + 1e-9:
                    # For tracking the first item, we only need to know which item was picked first
                    # If chosen is empty, this item is the first
                    first_item = idx if not chosen else chosen[0]
                    heapq.heappush(heap, (-take_ub, idx + 1, new_cap, new_val, [first_item]))
        
        # After B&B, find the best first item
        # We need to track which first items lead to good solutions
        # Simpler approach: for each candidate first item, compute the best total value
        # using greedy completion
        
        if time.time() < deadline:
            # Try each item as the first pick, then complete greedily by density
            for i in range(n_f):
                if w_s[i] > C + 1e-9:
                    continue
                if time.time() > deadline:
                    break
                
                # Pick item i first, then greedily fill rest by density
                rem_cap = C - w_s[i]
                total_val = v_s[i]
                
                for j in range(n_f):
                    if j == i:
                        continue
                    if w_s[j] <= rem_cap + 1e-9:
                        total_val += v_s[j]
                        rem_cap -= w_s[j]
                        if rem_cap <= 1e-9:
                            break
                
                if total_val > best_score:
                    best_score = total_val
                    best_i_local = i

    if best_i_local == -1:
        return -1
    
    return int(feasible_indices[best_i_local])
