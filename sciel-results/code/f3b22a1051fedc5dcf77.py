import time
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin for 60s limit
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 1
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    nv = len(valid_nodes)
    
    if nv == 0:
        return [1]
        
    # Build conflict graph using bitsets
    # conflicts[i] is a bitmask of valid nodes that conflict with valid_nodes[i]
    conflicts = [0] * nv
    for i_idx in range(nv):
        v = valid_nodes[i_idx]
        for i_j in range(i_idx + 1, nv):
            u = valid_nodes[i_j]
            if v + u in squares:
                conflicts[i_idx] |= (1 << i_j)
                conflicts[i_j] |= (1 << i_idx)
                
    def popcount(x):
        return bin(x).count('1')
        
    def mask_to_list(mask):
        res = []
        for i in range(nv):
            if mask & (1 << i):
                res.append(valid_nodes[i])
        return res
        
    def greedy_mis(order):
        chosen_mask = 0
        for idx in order:
            if (conflicts[idx] & chosen_mask) == 0:
                chosen_mask |= (1 << idx)
        return chosen_mask
        
    def local_search(mask, max_iters=1000):
        mask = mask
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
                
            # Try adding each non-chosen node
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
                    
            # Try 1-for-1 swaps
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    mask |= (1 << i)
                    improved = True
                    continue
                # Try removing each bit in conf_in_mask
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    # Check if removing r_idx allows adding i
                    # conflicts[i] & (mask ^ r_bit) == 0
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit
                        
            # Try 2-for-1 swaps (less frequent)
            if iters % 10 == 0:
                for i in range(nv):
                    if mask & (1 << i):
                        continue
                    conf_in_mask = conflicts[i] & mask
                    if conf_in_mask == 0:
                        mask |= (1 << i)
                        improved = True
                        continue
                    # Get conflicting indices
                    conf_list = []
                    temp = conf_in_mask
                    while temp:
                        r_bit = temp & (-temp)
                        r_idx = r_bit.bit_length() - 1
                        conf_list.append(r_idx)
                        temp ^= r_bit
                        
                    # Try removing pairs
                    for a_idx in range(len(conf_list)):
                        for b_idx in range(a_idx + 1, len(conf_list)):
                            a = conf_list[a_idx]
                            b = conf_list[b_idx]
                            a_bit = 1 << a
                            b_bit = 1 << b
                            new_mask = mask & ~a_bit & ~b_bit
                            if (conflicts[i] & new_mask) == 0:
                                mask = new_mask | (1 << i)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        return mask

    best_mask = 0
    best_count = 0
    
    # 1. Greedy by degree (ascending)
    order = sorted(range(nv), key=lambda i: popcount(conflicts[i]))
    mask = greedy_mis(order)
    mask = local_search(mask, 500)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
        
    # 2. Greedy by degree (descending)
    order = sorted(range(nv), key=lambda i: popcount(conflicts[i]), reverse=True)
    mask = greedy_mis(order)
    mask = local_search(mask, 500)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
        
    # 3. Random restarts
    import random
    random.seed(42)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Estimate iterations per second roughly? Just loop until time runs out.
        while time.time() < deadline - 2:
            order = list(range(nv))
            random.shuffle(order)
            mask = greedy_mis(order)
            # Shorter local search for random restarts
            mask = local_search(mask, 200)
            c = popcount(mask)
            if c > best_count:
                best_count = c
                best_mask = mask
                
    # Final Polish on best
    if time.time() < deadline - 1:
        mask = local_search(best_mask, 2000)
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
            
    result = mask_to_list(best_mask)
    
    # Verification
    res_set = set(result)
    for a in res_set:
        if 2 * a in squares:
            # Should not happen due to valid_nodes filter, but just in case
            res_set.remove(a)
            continue
        for b in res_set:
            if a <= b and (a + b) in squares:
                # Invalid pair found, remove one (this shouldn't happen if logic is correct)
                res_set.remove(b)
                break
                
    if not res_set:
        return [valid_nodes[0]]
        
    return list(res_set)
