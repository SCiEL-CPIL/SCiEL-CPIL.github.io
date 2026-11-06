import math

# EVOLVE-BLOCK-START
def construct():
    # Candidate vertices: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    
    # Map candidates to indices for faster graph operations
    idx_map = {v: i for i, v in enumerate(candidates)}
    n = len(candidates)
    
    # Build adjacency list as bitsets (integers) for efficiency
    # adj[i] is an integer with bit j set if candidates[i] and candidates[j] are connected
    adj = [0] * n
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Find the index of 255
    idx_255 = idx_map[255]
    
    # We want the max clique containing 255.
    # Initialize Bron-Kerbosch with R={255}, P=neighbors of 255 in candidates\{255}, X=empty
    
    # Represent P, X, and R as bitsets
    # Bit i is set if candidate i is in the set
    all_bits = (1 << n) - 1
    # P initially contains all candidates except 255
    P = all_bits & ~(1 << idx_255)
    X = 0
    R = 1 << idx_255
    
    max_clique_size = 1
    best_R_mask = R
    
    def bronk(R_mask, R_size, P_mask, X_mask):
        nonlocal max_clique_size, best_R_mask
        
        # If P and X are empty, we have a maximal clique
        if P_mask == 0 and X_mask == 0:
            if R_size > max_clique_size:
                max_clique_size = R_size
                best_R_mask = R_mask
            return
        
        if P_mask == 0:
            return
        
        # Pivot selection
        # Choose u from P U X that maximizes |P \cap N(u)|
        PX = P_mask | X_mask
        if PX == 0:
            return
            
        best_u = -1
        max_count = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp -= lsb
            
            # Count bits in P \cap N(u)
            count = bin(P_mask & adj[u]).count('1')
            if count > max_count:
                max_count = count
                best_u = u
        
        if best_u == -1:
            return
            
        # Iterate over v in P \ N(best_u)
        candidates_for_expansion = P_mask & ~adj[best_u]
        
        temp = candidates_for_expansion
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp -= lsb
            
            # Recurse
            new_R = R_mask | lsb
            new_P = P_mask & adj[v]
            new_X = X_mask & adj[v]
            
            bronk(new_R, R_size + 1, new_P, new_X)
            
            # Update P and X for the next iteration
            P_mask = P_mask & ~lsb
            X_mask = X_mask | lsb
            
    bronk(R, 1, P, X)
    
    # Reconstruct the best clique
    best_clique = []
    temp = best_R_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        best_clique.append(candidates[idx])
        temp -= lsb
        
    return best_clique

# EVOLVE-BLOCK-END
