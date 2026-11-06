import time
import sys
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Precompute compatibility
    # compat[i][j] is True if lcm(vertices[i], vertices[j]) > N
    # We can store this as a list of bitmasks for efficiency
    compat = [0] * n
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            # lcm(a, b) = a * b // gcd(a, b)
            g = gcd(a, b)
            if a * b // g > N:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    weights = [1.0 / v for v in vertices]
    
    # Order vertices by weight descending (index 0 is vertex 2, index 1 is vertex 3, etc.)
    # Since weights are 1/v, larger v means smaller weight.
    # So we want to consider small v first.
    # The vertices are already 2..256, so index 0 is 2, index 1 is 3...
    # This is already sorted by weight descending.
    order = list(range(n))
    
    # Greedy Construction
    # S is a bitmask of selected vertices
    S = 0
    for v in order:
        # Check if v is compatible with all in S
        # S & ~compat[v] should be 0
        if (S & ~compat[v]) == 0:
            S |= (1 << v)
            
    def get_value(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            s += weights[idx]
            m ^= lsb
        return s
        
    best_mask = S
    best_val = get_value(S)
    
    # Local Search: Remove and Refill
    # We iterate until no improvement or time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each vertex in the current best set
        # We need to iterate over the set bits of best_mask
        m = best_mask
        while m:
            lsb = m & -m
            u_idx = (lsb.bit_length() - 1)
            m ^= lsb
            
            # Candidate set without u
            S_candidate = best_mask & ~lsb
            val_candidate = best_val - weights[u_idx]
            
            # Greedy Refill
            # Try to add vertices from 'order' that are not in S_candidate
            # and are compatible with S_candidate
            # To make it faster, we can just check all vertices in order
            new_mask = S_candidate
            new_val = val_candidate
            
            for v in order:
                if new_mask & (1 << v):
                    continue
                # Check compatibility
                if (new_mask & ~compat[v]) == 0:
                    new_mask |= (1 << v)
                    new_val += weights[v]
                    
            if new_val > best_val:
                best_val = new_val
                best_mask = new_mask
                improved = True
                break # Restart the outer loop with the new best
        
        # If no improvement was found in the "remove" phase, try "add" phase?
        # The greedy refill in the remove phase effectively adds vertices.
        # If we didn't remove anything, we might miss adding a vertex that was skipped
        # because it conflicted with a vertex we *kept*.
        # But if we kept it, it's already in.
        # The only way to add a vertex is if it was previously blocked by some u.
        # Removing u allows it to be added.
        # So "Remove and Refill" covers "Add" moves implicitly if the blocking vertex is removed.
        # However, if adding a vertex w requires removing u, and w has higher weight than u?
        # The refill will add w if it's compatible.
        
    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = (lsb.bit_length() - 1)
        result.append(vertices[idx])
        m ^= lsb
        
    if not result:
        return [2]
        
    return result
