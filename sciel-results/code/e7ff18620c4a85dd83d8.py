import time
import math
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
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
    
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    node_set = set(valid_nodes)
    nv = len(valid_nodes)
    
    # Build conflict graph using bitsets for speed
    # conflicts[i] is a bitmask of valid nodes that conflict with valid_nodes[i]
    conflicts = [0] * nv
    for i_idx in range(nv):
        v = valid_nodes[i_idx]
        for i_j in range(i_idx + 1, nv):
            u = valid_nodes[i_j]
            if v + u in squares:
                conflicts[i_idx] |= (1 << i_j)
                conflicts[i_j] |= (1 << i_idx)
    
    # Precompute degrees
    def popcount(x):
        return bin(x).count('1')
    
    degrees = [popcount(conflicts[i]) for i in range(nv)]
    
    # --- Heuristic MIS with improved local search ---
    
    def greedy_mis_bitset(order_indices):
        chosen_mask = 0
        for idx in order_indices:
            # Check if idx conflicts with any chosen
            if (conflicts[idx] & chosen_mask) == 0:
                chosen_mask |= (1 << idx)
        return chosen_mask
    
    def mask_to_set(mask):
        s = set()
        for i in range(nv):
            if mask & (1 << i):
                s.add(valid_nodes[i])
        return s
    
    def local_search_bitset(mask, max_iters=3000):
        mask = mask
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 2:
                break
            
            # Try adding each non-chosen node
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
            
            # Try swaps: for each non-chosen node i, try removing one conflict
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    # Can add directly
                    mask |= (1 << i)
                    improved = True
                    continue
                # Try removing each bit in conf_in_mask
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    # Check if i conflicts with any other node in mask besides r_idx
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit
        
        return mask
    
    def local_search_2swap(mask, max_iters=2000):
        """Try 2-for-1 and 1-for-2 swaps for better improvement"""
        mask = mask
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 2:
                break
            
            # Try adding each non-chosen node
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
            
            # Try 1-for-1 swaps (remove one, add one)
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    mask |= (1 << i)
                    improved = True
                    continue
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit
            
            # Try 1-for-2 swaps: remove two conflicting nodes, add one that was blocked
            # This is more expensive so do it less frequently
            if iters % 5 == 0:
                non_chosen = []
                for i in range(nv):
                    if not (mask & (1 << i)):
                        non_chosen.append(i)
                
                for i in non_chosen[:50]:  # Limit for speed
                    conf_in_mask = conflicts[i] & mask
                    if conf_in_mask == 0:
                        mask |= (1 << i)
                        improved = True
                        continue
                    # Try removing pairs from conf_in_mask
                    conf_bits = []
                    temp = conf_in_mask
                    while temp:
                        b = temp & (-temp)
                        conf_bits.append(b.bit_length() - 1)
                        temp ^= b
                    
                    for a_idx in range(len(conf_bits)):
                        for b_idx in range(a_idx + 1, len(conf_bits)):
                            a = conf_bits[a_idx]
                            b = conf_bits[b_idx]
                            # Check if removing both a and b allows adding i
                            new_mask = mask & ~(1 << a) & ~(1 << b)
                            if (conflicts[i] & new_mask) == 0:
                                # Also need to check that a and b don't have other conflicts
                                # that would be exposed - but since we're removing them,
                                # we just need i to be compatible with the rest
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
    
    random.seed(42)
    
    # Strategy 1: Min-degree greedy
    order_by_degree = sorted(range(nv), key=lambda i: (degrees[i], i))
    mask = greedy_mis_bitset(order_by_degree)
    mask = local_search_bitset(mask, max_iters=2000)
    mask = local_search_2swap(mask, max_iters=1000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
    
    # Strategy 2: Min-degree with random tie-breaking
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        order = sorted(range(nv), key=lambda i: (degrees[i], random.random()))
        mask = greedy_mis_bitset(order)
        mask = local_search_bitset(mask, max_iters=1500)
        mask = local_search_2swap(mask, max_iters=500)
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
    
    # Strategy 3: Random restarts
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(nv))
        random.shuffle(order)
        mask = greedy_mis_bitset(order)
        mask = local_search_bitset(mask, max_iters=1000)
        mask = local_search_2swap(mask, max_iters=500)
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
    
    # Strategy 4: Start from best and do more local search
    if time.time() < deadline - 3:
        mask = best_mask
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            mask = local_search_bitset(mask, max_iters=500)
            mask = local_search_2swap(mask, max_iters=300)
            c = popcount(mask)
            if c > best_count:
                best_count = c
                best_mask = mask
    
    # Final: try to add any remaining nodes
    mask = best_mask
    for i in range(nv):
        if not (mask & (1 << i)) and (conflicts[i] & mask) == 0:
            mask |= (1 << i)
    best_mask = mask
    best_count = popcount(best_mask)
    
    # Convert to list of nodes
    result = mask_to_set(best_mask)
    if not result:
        result = {valid_nodes[0]}
    
    # Verify
    res_set = set(result)
    for a in res_set:
        if 2 * a in squares:
            result = [valid_nodes[0]]
            break
        for b in res_set:
            if a <= b and (a + b) in squares:
                result = [valid_nodes[0]]
                break
        if result != list(res_set):
            break
    
    return list(result)
