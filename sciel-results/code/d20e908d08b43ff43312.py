import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    N = 256
    
    # Precompute 4-APs
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Build conflict graph: two elements conflict if they appear in the same 4-AP
    # Use adjacency as sets for fast lookup
    conflict = [set() for _ in range(N + 1)]
    for ap in all_4aps:
        for i in range(4):
            for j in range(i + 1, 4):
                conflict[ap[i]].add(ap[j])
                conflict[ap[j]].add(ap[i])
    
    # Precompute 3-APs
    all_3aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            all_3aps.append((a, a + d, a + 2 * d))
            
    # Map element to list of 3-AP indices it belongs to
    elem_to_3aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_3aps):
        for x in ap:
            elem_to_3aps[x].append(idx)

    # State: use integer bitmask for fast operations
    A_mask = 0
    current_3ap_count = 0
    
    def in_A(x):
        return (A_mask >> (x - 1)) & 1
    
    def add_to_A(x):
        nonlocal A_mask
        A_mask |= (1 << (x - 1))
    
    def remove_from_A(x):
        nonlocal A_mask
        A_mask &= ~(1 << (x - 1))
    
    def is_valid_add(x):
        """Check if adding x keeps set 4-AP free. Assumes x NOT in A."""
        for y in conflict[x]:
            if in_A(y):
                return False
        return True
    
    def is_valid_swap(x, y):
        """Check if swapping x(out) and y(in) keeps set 4-AP free. x in A, y not in A."""
        for z in conflict[y]:
            if z != x and in_A(z):
                return False
        return True
    
    def gain_add(x):
        """Gain in 3-AP count if we add x to A. Assumes x NOT in A."""
        gain = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for y in ap:
                if y != x and in_A(y):
                    cnt += 1
            if cnt == 2:
                gain += 1
        return gain
    
    def loss_remove(x):
        """Loss in 3-AP count if we remove x from A. Assumes x IS in A."""
        loss = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for y in ap:
                if y != x and in_A(y):
                    cnt += 1
            if cnt == 2:
                loss += 1
        return loss
    
    def delta_swap(x, y):
        """Change in 3-AP count if we swap x(out) and y(in). x in A, y not in A."""
        loss = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for z in ap:
                if z != x and in_A(z):
                    cnt += 1
            if cnt == 2:
                loss += 1
                
        gain = 0
        for idx in elem_to_3aps[y]:
            ap = all_3aps[idx]
            cnt = 0
            for z in ap:
                if z != y and in_A(z) and z != x:
                    cnt += 1
            if cnt == 2:
                gain += 1
                
        return gain - loss
    
    # Greedy Initialization with multiple restarts using different orderings
    best_set_mask = 0
    best_score = 0
    
    for restart in range(5):
        A_mask = 0
        current_3ap_count = 0
        remaining = set(range(1, N + 1))
        
        # Shuffle remaining for diversity
        rem_list = list(remaining)
        random.shuffle(rem_list)
        
        for x in rem_list:
            if x not in remaining:
                continue
            if is_valid_add(x):
                g = gain_add(x)
                add_to_A(x)
                current_3ap_count += g
                remaining.remove(x)
        
        if current_3ap_count > best_score:
            best_score = current_3ap_count
            best_set_mask = A_mask
    
    # Restore best state
    A_mask = best_set_mask
    current_3ap_count = best_score
    
    # Branch and Bound for maximum weight independent set on conflict graph
    # Weights are the marginal 3-AP contributions (approximated)
    # Use a simplified B&B that works on the current solution as a seed
    
    def bb_upper_bound(candidates, current_count):
        """Upper bound: current_count + sum of max possible gains for each candidate."""
        ub = current_count
        for x in candidates:
            if is_valid_add(x):
                ub += max(0, gain_add(x))
        return ub
    
    # Improved local search with hill climbing and simulated annealing
    T_start = 15.0
    T_end = 0.1
    total_time = deadline - start_time
    
    while time.time() < deadline:
        # Time-based cooling
        elapsed = time.time() - start_time
        progress = min(elapsed / total_time, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        r = random.random()
        
        if r < 0.35:
            # Try Add
            x = random.randint(1, N)
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        add_to_A(x)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
        elif r < 0.55:
            # Try Remove
            x = random.randint(1, N)
            if in_A(x):
                delta = -loss_remove(x)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    remove_from_A(x)
                    current_3ap_count += delta
                    if current_3ap_count > best_score:
                        best_score = current_3ap_count
                        best_set_mask = A_mask
        else:
            # Try Swap
            x = random.randint(1, N)
            y = random.randint(1, N)
            if in_A(x) and not in_A(y):
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
    
    # Final local search polish: try all single-element swaps and adds/removes
    A_mask = best_set_mask
    current_3ap_count = best_score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding elements
        for x in range(1, N + 1):
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta > 0:
                        add_to_A(x)
                        current_3ap_count += delta
                        improved = True
        # Try removing elements
        for x in range(1, N + 1):
            if in_A(x):
                delta = -loss_remove(x)
                if delta > 0:
                    remove_from_A(x)
                    current_3ap_count += delta
                    improved = True
        # Try swaps - only do a limited number per iteration to save time
        active = [x for x in range(1, N + 1) if in_A(x)]
        inactive = [x for x in range(1, N + 1) if not in_A(x)]
        # Limit swap attempts
        max_swaps = 500
        swap_count = 0
        for x in active:
            for y in inactive:
                if swap_count >= max_swaps:
                    break
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta > 0:
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        improved = True
                        swap_count += 1
                        break
            if improved or swap_count >= max_swaps:
                break
    
    # Return best set found
    res = [i for i in range(1, N + 1) if (best_set_mask >> (i - 1)) & 1]
    
    if not res:
        return [1]
        
    return res
