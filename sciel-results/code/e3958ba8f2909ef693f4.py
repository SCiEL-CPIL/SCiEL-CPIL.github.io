import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    # 1. Precompute conflict bitsets
    # conflict[a][b] is a bitmask of c such that {a,b,c} is a bad triple
    # We use 1-indexed, so bit i corresponds to number i+1? 
    # Let's use 0-indexed for bitmasks. Bit i corresponds to number i+1.
    # So number k is represented by bit (k-1).
    
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    
    # Precompute LCMs
    lcms = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcms[i][j] = lcm(i, j)
            lcms[j][i] = lcms[i][j]

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcms[a][b]
            for c in range(b + 1, N + 1):
                if lcms[a][c] == lab and lcms[b][c] == lab:
                    # {a,b,c} is a bad triple
                    # Set bits
                    conflict[a][b] |= (1 << (c - 1))
                    conflict[b][a] |= (1 << (c - 1))
                    
                    conflict[a][c] |= (1 << (b - 1))
                    conflict[c][a] |= (1 << (b - 1))
                    
                    conflict[b][c] |= (1 << (a - 1))
                    conflict[c][b] |= (1 << (a - 1))

    if time.time() > deadline:
        return list(range(1, N + 1)) # Fallback, though unlikely to reach here this fast

    def is_valid_add(v, selected_mask):
        """
        Check if adding v to the set represented by selected_mask is valid.
        v is 1-indexed.
        selected_mask is a bitmask of 1-indexed numbers.
        """
        # For every u in selected_mask, check if conflict[u][v] intersects with selected_mask
        # If yes, there is a w in selected_mask such that {u, v, w} is a bad triple.
        temp_mask = selected_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            u_idx = lsb.bit_length() - 1
            u = u_idx + 1
            
            # Check if conflict[u][v] has any common bits with selected_mask
            if conflict[u][v] & selected_mask:
                return False
            
            temp_mask ^= lsb
        return True

    def greedy_select(order):
        selected_mask = 0
        selected_list = []
        for v in order:
            if time.time() > deadline:
                break
            if is_valid_add(v, selected_mask):
                selected_mask |= (1 << (v - 1))
                selected_list.append(v)
        return selected_mask, selected_list

    def mask_to_list(mask):
        res = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res.append(idx + 1)
            m ^= lsb
        return res

    best_mask = 0
    best_size = 0
    
    # 1. Greedy with multiple orderings
    random.seed(42)
    
    # Order 1: Ascending value
    order1 = list(range(1, N + 1))
    m1, l1 = greedy_select(order1)
    if len(l1) > best_size:
        best_size = len(l1)
        best_mask = m1

    # Order 2: Descending value
    order2 = list(range(N, 0, -1))
    m2, l2 = greedy_select(order2)
    if len(l2) > best_size:
        best_size = len(l2)
        best_mask = m2

    # Order 3: Randomized Greedy with restarts
    for i in range(100):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        mi, li = greedy_select(order)
        if len(li) > best_size:
            best_size = len(li)
            best_mask = mi

    # 2. Local Search: 1-swap
    # Try to remove one element and add another (net 0 change, but might enable more adds?)
    # Or remove 1, add 1, then try to add more.
    # Actually, standard 1-swap: remove u, add v. Size stays same.
    # Then try to add more.
    
    def try_add_more(mask, current_list):
        # Try to add any remaining element
        remaining_mask = ((1 << N) - 1) & ~mask
        # Iterate through remaining
        temp = remaining_mask
        new_mask = mask
        new_list = list(current_list)
        improved = True
        while improved:
            improved = False
            temp = ((1 << N) - 1) & ~new_mask
            while temp:
                lsb = temp & -temp
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                if is_valid_add(v, new_mask):
                    new_mask |= lsb
                    new_list.append(v)
                    improved = True
                    # Restart scan as adding v might block others, but we just need to find *any* improvement
                    # Actually, greedy add: just keep adding if possible
                    break # Restart the inner loop? No, just continue to next candidate
                temp ^= lsb
        return new_mask, new_list

    # Apply try_add_more to best_mask
    best_list = mask_to_list(best_mask)
    best_mask, best_list = try_add_more(best_mask, best_list)
    best_size = len(best_list)

    # 3. Local Search: 1-for-1 swap + fill
    # For each u in best_set, for each v not in best_set:
    #   If removing u allows adding v (and v doesn't conflict with rest), do it.
    #   Then try to fill.
    
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_list = mask_to_list(best_mask)
            current_set = set(current_list)
            
            for u in current_list:
                if time.time() > deadline - 2:
                    break
                # Remove u
                mask_without_u = best_mask & ~(1 << (u - 1))
                
                # Try to add any v not in current_set
                remaining = ((1 << N) - 1) & ~mask_without_u
                temp = remaining
                found_swap = False
                while temp:
                    lsb = temp & -temp
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    if v not in current_set:
                        if is_valid_add(v, mask_without_u):
                            # Perform swap
                            new_mask = mask_without_u | lsb
                            new_list = [x for x in current_list if x != u]
                            new_list.append(v)
                            
                            # Try to fill
                            new_mask, new_list = try_add_more(new_mask, new_list)
                            
                            if len(new_list) > best_size:
                                best_size = len(new_list)
                                best_mask = new_mask
                                improved = True
                                found_swap = True
                                break
                    temp ^= lsb
                if found_swap:
                    break # Restart outer loop with new best

    return mask_to_list(best_mask)
