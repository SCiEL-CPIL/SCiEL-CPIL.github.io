"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_clique = []
    best_size = 0
    
    def expand(R, P_mask):
        nonlocal best_clique, best_size
        
        # Bound: if current size + number of candidates <= best size, prune
        if len(R) + P_mask.bit_count() <= best_size:
            return
            
        if P_mask == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # Pick a pivot vertex u from P_mask
        # Choose one with max degree in P_mask to maximize pruning
        u = -1
        max_deg = -1
        temp_mask = P_mask
        while temp_mask:
            v = (temp_mask & -temp_mask).bit_length() - 1
            temp_mask &= temp_mask - 1
            deg = (adj[v] & P_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                u = v
                
        # Iterate over vertices in P_mask that are NOT neighbors of u
        # This reduces the number of branches
        candidates_mask = P_mask & ~(adj[u] | (1 << u))
        
        # If candidates_mask is 0, we must process all in P_mask?
        # No, the property is that a max clique must contain a vertex from P_mask \ N(u)
        # Wait, if P_mask \ N(u) is empty, then P_mask is a clique? No.
        # The theorem says: There exists a max clique in R ∪ (P \ N(u)).
        # So we only need to branch on P \ N(u).
        # But if P \ N(u) is empty, it means P is a clique? No, it means u is connected to all in P.
        # In that case, we can just pick any v in P.
        # Actually, if P \ N(u) is empty, then for all v in P, v is in N(u).
        # Then we can just branch on any v in P.
        
        if candidates_mask == 0:
             # Pick any vertex from P_mask
             v = (P_mask & -P_mask).bit_length() - 1
             new_R = R + [v]
             new_P_mask = P_mask & adj[v] & ~(1 << v)
             expand(new_R, new_P_mask)
             return

        # Process candidates in some order (e.g., descending degree)
        # We will iterate through the bits of candidates_mask
        # To order them, we can collect them and sort, or just iterate.
        # For N=24, sorting is cheap.
        cand_list = []
        temp = candidates_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            cand_list.append(v)
            
        # Sort by degree in P_mask descending
        cand_list.sort(key=lambda x: (adj[x] & P_mask).bit_count(), reverse=True)
        
        for v in cand_list:
            new_R = R + [v]
            # New P is neighbors of v in P_mask
            new_P_mask = P_mask & adj[v] & ~(1 << v)
            expand(new_R, new_P_mask)
            
            # Remove v from P_mask for subsequent iterations?
            # In the standard Bron-Kerbosch with pivot, we don't explicitly remove v from P_mask in the loop
            # because the next candidate is chosen from the original P_mask \ N(u).
            # However, the recursive call handles the restriction.
            # But we must ensure we don't pick the same v again.
            # The loop iterates over `cand_list`, which is fixed.
            # This is correct.
            
            # Early termination check
            if len(R) + (P_mask & ~(1 << v)).bit_count() <= best_size:
                 # This bound is slightly off because P_mask changes in recursion,
                 # but for the current level, if we remove v, the remaining candidates
                 # cannot form a clique larger than best_size - len(R).
                 # Actually, the bound at the start of expand is sufficient.
                 # We can break if the remaining candidates in cand_list are too few.
                 pass

    # Initial call
    all_mask = (1 << n) - 1
    expand([], all_mask)
    
    if best_size < 2:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    branches = best_clique
    paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            paths.append([branches[i], branches[j]])
            
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
