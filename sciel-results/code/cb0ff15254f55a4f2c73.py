import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    # 1. Generate candidate set: numbers in [1,255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    
    # 2. Build adjacency as bitmasks for fast operations
    # adj_mask[i] is a bitmask where bit j is set if gcd(candidates[i], candidates[j]) > 1
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Precompute: for each vertex, the mask of all other vertices (full mask minus self)
    full_mask = (1 << n) - 1
    
    # 3. Bron-Kerbosch with bitmasks and pivoting
    max_clique_mask = 0
    max_size = 0
    
    # Use a list to track the best clique for reconstruction
    best_clique_list = []
    
    def popcount(x):
        return bin(x).count('1')
    
    def bronk(R_mask, P_mask, X_mask):
        nonlocal max_clique_mask, max_size, best_clique_list
        
        # Pruning: if current + potential <= best, return
        if popcount(R_mask) + popcount(P_mask) <= max_size:
            return
        
        if P_mask == 0 and X_mask == 0:
            if popcount(R_mask) > max_size:
                max_size = popcount(R_mask)
                max_clique_mask = R_mask
                # Extract the list
                best_clique_list = []
                for i in range(n):
                    if R_mask & (1 << i):
                        best_clique_list.append(i)
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        # We want u that maximizes |P ∩ N(u)|, so we iterate over P \ N(u)
        PX_mask = P_mask | X_mask
        
        # Find pivot: vertex in PX with max |P & adj_mask[u]|
        best_u = -1
        best_deg = -1
        temp = PX_mask
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            u = lb.bit_length() - 1
            temp ^= lb
            
            deg = popcount(P_mask & adj_mask[u])
            if deg > best_deg:
                best_deg = deg
                best_u = u
        
        if best_u == -1:
            # No pivot, iterate over all of P
            to_iterate = P_mask
        else:
            # Iterate over P \ N(u)
            to_iterate = P_mask & ~adj_mask[best_u]
        
        # Iterate over vertices in to_iterate
        temp = to_iterate
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            temp ^= lb
            
            # Check time
            if time.time() > deadline:
                return
            
            new_R = R_mask | (1 << v)
            new_P = P_mask & adj_mask[v]
            new_X = X_mask & adj_mask[v]
            
            bronk(new_R, new_P, new_X)
            
            # Move v from P to X
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
            
            # Update to_iterate for remaining iterations
            # (not strictly necessary since we're using a separate temp, but safe)
    
    # Initial call
    bronk(0, (1 << n) - 1, 0)
    
    # If no clique found or timeout, fallback to multiples of 3
    if max_size == 0 or not best_clique_list:
        return [i for i in range(1, 256) if i % 3 == 0]
    
    # Convert indices to values
    result = [candidates[i] for i in best_clique_list]
    
    # Ensure 255 is in the result
    if 255 not in result:
        # Find index of 255 in candidates
        idx_255 = candidates.index(255)
        # Check if we can add it (should already be there if it's in the max clique)
        # If not, we may need to handle this edge case
        # But since 255 shares factors with all candidates (div by 3,5,17), 
        # it should be connected to all, so it should be in any max clique
        # If somehow it's not, add it and verify
        result.append(255)
        # Verify pairwise gcd
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback
            return [i for i in range(1, 256) if i % 3 == 0]
    
    return result
