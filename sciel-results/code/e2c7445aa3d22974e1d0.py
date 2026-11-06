import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # Candidates must share a factor with 255
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in candidates (it is, 255%3==0)
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency bitmasks
    # adj[i] is an integer where bit j is 1 if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy initialization: Multiples of 3 form a clique
    # This gives us a baseline of size 85
    best_clique_indices = [i for i, c in enumerate(candidates) if c % 3 == 0]
    best_size = len(best_clique_indices)
    
    # Bron-Kerbosch with bitmasks
    # R: list of indices in current clique
    # P: bitmask of candidates to add
    # X: bitmask of candidates already processed (for maximal clique property)
    # We force 255 to be in the clique, so we start with R=[255], P=N(255), X=0
    
    def popcount(x):
        return x.bit_count()
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique_indices, best_size
        
        if time.time() > deadline:
            return
        
        # Update best if current R is larger
        if len(R) > best_size:
            best_clique_indices = list(R)
            best_size = len(R)
        
        if P == 0 and X == 0:
            return
        
        # Choose pivot u from P | X that maximizes |P & N(u)|
        # If P is empty, we can't pick a pivot from P, but X might have elements.
        # Standard BK: if P is empty, return.
        # Here we need to be careful. If P is empty, no more vertices can be added to R.
        # The clique R is maximal if X is also empty? No, R is maximal in the graph.
        # But we only care about size.
        
        if P == 0:
            return
            
        pivot_set = P | X
        best_u = -1
        max_count = -1
        
        # Iterate through bits in pivot_set to find u
        # Optimization: Just iterate through all vertices in P|X?
        # n is small (127).
        for u in range(n):
            if (pivot_set >> u) & 1:
                count = popcount(P & adj[u])
                if count > max_count:
                    max_count = count
                    best_u = u
        
        # Iterate over candidates in P \ N(u)
        # These are the vertices v in P such that v is NOT adjacent to u
        candidates_to_process = P & ~adj[best_u]
        
        while candidates_to_process:
            if time.time() > deadline:
                return
            
            # Pick a vertex v from candidates_to_process
            # We can pick the lowest set bit
            lsb = candidates_to_process & -candidates_to_process
            v = lsb.bit_length() - 1
            
            # Remove v from candidates_to_process
            candidates_to_process ^= lsb
            
            # Recurse
            new_R = R + [v]
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Remove v from P and add to X
            P &= ~lsb
            X |= lsb
            
            # If P is empty, we can break early? 
            # No, we need to continue the while loop on candidates_to_process?
            # Actually, if P becomes 0, the recursive call would return immediately.
            # But we still need to update X for the other vertices in candidates_to_process?
            # No, the loop structure of BK is:
            # for v in P \ N(u):
            #   ...
            #   P = P \ {v}
            #   X = X U {v}
            # So we must continue iterating over the original set P \ N(u).
            # But note: P is being modified.
            # The set `candidates_to_process` was computed from the initial P.
            # As we remove vertices from P, they might not be in `candidates_to_process` anymore?
            # No, `candidates_to_process` is a snapshot of `P & ~adj[u]`.
            # We iterate through this snapshot.
            # Inside the loop, we update P and X.
            
    # Start search
    # R = [255]
    # P = neighbors of 255
    # X = 0
    P = adj[idx_255]
    X = 0
    R = [idx_255]
    
    bron_kerbosch(R, P, X)
    
    # Convert back to values
    result = [candidates[i] for i in best_clique_indices]
    result.sort()
    
    return result
