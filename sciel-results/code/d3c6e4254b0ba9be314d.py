import random
from itertools import combinations
from math import lcm, gcd
from functools import lru_cache

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute conflicts
    # conflicts[i][j] is a bitmask of k such that (i, j, k) is a forbidden triple
    # Note: (i,j,k) forbidden iff lcm(i,j)==lcm(i,k)==lcm(j,k)
    # We store this in a 2D list of ints (bitmasks)
    conflicts = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a, b, c in combinations(numbers, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            # Mark c in conflicts[a][b], conflicts[b][a]
            # Mark b in conflicts[a][c], conflicts[c][a]
            # Mark a in conflicts[b][c], conflicts[c][b]
            conflicts[a][b] |= (1 << c)
            conflicts[b][a] |= (1 << c)
            
            conflicts[a][c] |= (1 << b)
            conflicts[c][a] |= (1 << b)
            
            conflicts[b][c] |= (1 << a)
            conflicts[c][b] |= (1 << a)
            
    # Static degree: how many conflicts does i have with others?
    # This helps when S is empty
    static_degree = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i != j:
                static_degree[i] += bin(conflicts[i][j]).count('1')
                # Actually, conflicts[i][j] contains k. 
                # The "degree" of i in the hypergraph is the number of pairs (j,k) it participates in.
                # This sum counts pairs (j,k) such that (i,j,k) is forbidden.
                # This is a good measure of "badness".

    best_size = 0
    best_set = []
    
    # Initial C mask: all numbers 1..128
    # Bit 0 is unused, bits 1..128 used
    initial_C_mask = (1 << (N + 1)) - 2 # Bits 1..128 set
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
        
    def search(S_list, C_mask, current_size):
        nonlocal best_size, best_set
        
        if current_size + popcount(C_mask) <= best_size:
            return
            
        if C_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(S_list)
            return
            
        # Heuristic: Pick x from C that maximizes pruning
        # If S is empty, use static_degree
        best_x = -1
        max_prune = -1
        
        # Iterate through bits in C_mask
        cm = C_mask
        candidates = []
        while cm:
            # Get lowest set bit
            lsb = cm & -cm
            x = lsb.bit_length() - 1
            candidates.append(x)
            cm ^= lsb
            
        for x in candidates:
            # Calculate blocked mask if we add x
            blocked = 0
            for a in S_list:
                blocked |= conflicts[a][x]
            
            # The new C would be (C_mask without x) and without blocked
            # Prune amount is number of bits in (C_mask without x) that are in blocked
            # Note: x itself is in C_mask. We remove it.
            # blocked might contain x? No, distinct elements.
            # blocked might contain elements not in C_mask? Yes.
            # We only care about bits in C_mask.
            
            # Potential new C mask:
            # new_C = (C_mask & ~(1 << x)) & ~blocked
            # Removed = (C_mask & ~(1 << x)) & blocked
            removed = (C_mask & ~(1 << x)) & blocked
            prune_count = popcount(removed)
            
            if prune_count > max_prune:
                max_prune = prune_count
                best_x = x
                
        # If S is empty, prune_count is 0 for all. 
        # In that case, pick x with max static_degree
        if max_prune == 0 and len(S_list) == 0:
             for x in candidates:
                 if static_degree[x] > static_degree[best_x]:
                     best_x = x
                     
        # Branch 1: Include best_x
        x = best_x
        new_S = S_list + [x]
        blocked = 0
        for a in S_list:
            blocked |= conflicts[a][x]
        new_C_mask = (C_mask & ~(1 << x)) & ~blocked
        search(new_S, new_C_mask, current_size + 1)
        
        # Branch 2: Exclude best_x
        search(S_list, C_mask & ~(1 << x), current_size)

    search([], initial_C_mask, 0)
    
    if not best_set:
        return [1]
        
    return best_set

