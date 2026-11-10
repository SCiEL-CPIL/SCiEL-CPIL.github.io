import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # Build adjacency masks
    # adj_mask[i] is an integer where bit j is 1 if i and j are connected (lcm(i,j) <= n)
    adj_mask = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
        if time.time() > deadline:
            # Fallback
            return [1, 2]

    # Initial best clique: Divisors of 504
    best_mask = 0
    best_size = 0
    for k in range(1, 513):
        if 504 % k == 0:
            best_mask |= (1 << k)
            best_size += 1
            
    # Helper to count bits
    # Python 3.10+ has int.bit_count()
    # Fallback if needed
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_mask, best_size
        
        if time.time() > deadline:
            return

        # If P is empty
        if P_mask == 0:
            if X_mask == 0:
                r_size = popcount(R_mask)
                if r_size > best_size:
                    best_size = r_size
                    best_mask = R_mask
            return

        # Pruning: if max possible size <= current best, return
        # Max possible size is len(R) + len(P)
        if popcount(R_mask) + popcount(P_mask) <= best_size:
            return

        # Pivot selection: u in P U X maximizing popcount(P & adj_mask[u])
        # Usually picking from P is sufficient if P is not empty.
        # To be safe and fast, we can just pick from P.
        # If P is empty, we returned.
        
        # Find pivot
        pivot = 0
        max_count = -1
        # Iterate through bits in P ∪ X
        temp_p = P_mask | X_mask
        while temp_p:
            # Get lowest set bit
            bit = temp_p & -temp_p
            temp_p ^= bit
            u = bit.bit_length() - 1 # Index of the bit
            
            count = popcount(P_mask & adj_mask[u])
            if count > max_count:
                max_count = count
                pivot_idx = u
        
        # If no pivot found (should not happen if P not empty), find any in P
        if pivot_idx == 0:
            bit = P_mask & -P_mask
            pivot_idx = bit.bit_length() - 1
            
        # Candidates: P \ N(pivot)
        # N(pivot) is adj_mask[pivot_idx]
        # We want vertices in P that are NOT connected to pivot.
        # candidates_mask = P_mask & ~adj_mask[pivot_idx]
        
        candidates_mask = P_mask & ~adj_mask[pivot_idx]
        
        # If candidates is empty, we can return.
        # Why? Because all vertices in P are connected to pivot.
        # Any clique formed from P can be extended by pivot?
        # Standard BK logic: If P \ N(u) is empty, return.
        if candidates_mask == 0:
            return

        # Iterate through candidates
        # We need to process each v in candidates_mask
        # For each v:
        #   R' = R U {v}
        #   P' = P \ N(v)
        #   X' = X \ N(v)
        
        temp_cand = candidates_mask
        while temp_cand:
            bit = temp_cand & -temp_cand
            temp_cand ^= bit
            v = bit.bit_length() - 1
            
            # Recurse
            R_new = R_mask | bit
            P_new = P_mask & adj_mask[v]
            X_new = X_mask & adj_mask[v]
            
            bron_kerbosch(R_new, P_new, X_new)
            
            # Update P and X for next iteration
            # P = P \ {v}
            P_mask ^= bit
            # X = X U {v}
            X_mask |= bit
            
            if time.time() > deadline:
                return

    # Initial call
    # R = 0
    # P = all nodes (bits 1 to 512)
    # X = 0
    
    # Create mask for all nodes 1..512
    all_nodes_mask = 0
    for i in range(1, n + 1):
        all_nodes_mask |= (1 << i)
        
    bron_kerbosch(0, all_nodes_mask, 0)
    
    # Reconstruct list from best_mask
    result = []
    temp_res = best_mask
    while temp_res:
        bit = temp_res & -temp_res
        temp_res ^= bit
        result.append(bit.bit_length() - 1)
        
    if not result:
        return [1]
        
    return result
