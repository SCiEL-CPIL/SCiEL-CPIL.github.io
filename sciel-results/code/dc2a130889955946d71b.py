import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Finds the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # 1. Generate candidate vertices
    # Vertices must share a factor with 255 (3, 5, 17)
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    
    # 2. Build adjacency using bitmasks for speed
    # adj_mask[i] is a bitmask where bit j is set if gcd(candidates[i], candidates[j]) > 1
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    best_clique = []
    
    def bron_kerbosch(R, P_mask, X_mask):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if P_mask == 0 and X_mask == 0:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
            
        # Choose pivot u from P U X to minimize branching
        # Heuristic: pick u with max degree in P
        U_mask = P_mask | X_mask
        if U_mask:
            # Find pivot with max neighbors in P
            best_pivot = -1
            best_deg = -1
            u = U_mask
            while u:
                lb = u & (-u)
                idx = lb.bit_length() - 1
                u ^= lb
                deg = (adj_mask[idx] & P_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_pivot = idx
            # Branch on vertices in P that are NOT neighbors of pivot
            candidates_mask = P_mask & ~(adj_mask[best_pivot])
        else:
            candidates_mask = P_mask
        
        while candidates_mask:
            if time.time() > deadline:
                return
            lb = candidates_mask & (-candidates_mask)
            v = lb.bit_length() - 1
            candidates_mask ^= lb
            
            new_R = R | {v}
            new_P_mask = P_mask & adj_mask[v]
            new_X_mask = X_mask & adj_mask[v]
            
            bron_kerbosch(new_R, new_P_mask, new_X_mask)
            
            P_mask ^= lb
            X_mask |= lb
    
    # Initial call
    all_mask = (1 << n) - 1
    bron_kerbosch(set(), all_mask, 0)
    
    # Map indices back to values
    if best_clique:
        result = [candidates[i] for i in best_clique]
        # Ensure 255 is included (it should be, but just in case)
        if 255 not in result:
             result.append(255)
        return sorted(result)
    else:
        # Fallback: multiples of 3
        return list(range(3, 256, 3))

# EVOLVE-BLOCK-END
