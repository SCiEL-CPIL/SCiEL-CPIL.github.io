import time
import random

def construct():
    """
    Constructs a graph with no K5 and no I5, maximizing the number of vertices.
    Returns (n, edges).
    """
    start_time = time.time()
    DEADLINE = 250  # Safety margin before 300s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency as list of bitmasks (Python ints)
    # adj[i] is a bitmask where bit j is 1 if (i, j) is an edge
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Helper to get current edges list
    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only count edges where i < j
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                mask ^= lsb
        return edges

    # Precompute 4-cliques and 4-independent sets
    # A 4-clique is a set of 4 vertices all connected to each other.
    # An 4-independent set is a set of 4 vertices with no edges between them.
    
    def find_4_cliques():
        cliques = []
        # Iterate over all combinations of 4 vertices
        # This is O(n^4) which is 37^4 ~ 1.8M, acceptable for initial setup.
        # For larger n, this becomes expensive, but we expect n to be small (38-40).
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)): continue
                for k in range(j+1, n):
                    if not (adj[i] & (1 << k)): continue
                    if not (adj[j] & (1 << k)): continue
                    for l in range(k+1, n):
                        if not (adj[i] & (1 << l)): continue
                        if not (adj[j] & (1 << l)): continue
                        if not (adj[k] & (1 << l)): continue
                        # Found a K4: {i, j, k, l}
                        # Represent as a bitmask
                        mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                        cliques.append(mask)
        return cliques

    def find_4_independent_sets():
        indep_sets = []
        # Independent set: no edges between vertices.
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j): continue
                for k in range(j+1, n):
                    if adj[i] & (1 << k): continue
                    if adj[j] & (1 << k): continue
                    for l in range(k+1, n):
                        if adj[i] & (1 << l): continue
                        if adj[j] & (1 << l): continue
                        if adj[k] & (1 << l): continue
                        # Found an I4: {i, j, k, l}
                        mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                        indep_sets.append(mask)
        return indep_sets

    k4_list = find_4_cliques()
    i4_list = find_4_independent_sets()
    
    best_n = n
    best_edges = get_edges()
    
    # Randomized search to add vertices
    # We try to add vertices one by one.
    # For each new vertex, we try random adjacency patterns.
    # We keep the best graph found.
    
    # To speed up, we can restrict the search space for the new vertex's adjacency.
    # A good heuristic is to try to make the new vertex have a "balanced" degree or mimic existing patterns.
    # However, random sampling with validation is robust.
    
    # Optimization: When checking a new vertex v with mask M:
    # It creates a K5 if there is a K4 C such that C subset M.
    # It creates an I5 if there is an I4 I such that I subset ~M.
    
    # We will try to extend the graph.
    # Since we can't easily "remove" a bad vertex in a simple greedy loop without backtracking,
    # we will use a "try and keep" approach for the current state.
    # If we get stuck, we might restart from a slightly different initial random seed or just return best.
    # Given the time limit, a single pass with many random attempts for each new vertex is best.
    
    max_attempts_per_vertex = 1000
    target_n = 45 # Upper bound known is 46, let's aim for 40+
    
    while n < target_n and time.time() - start_time < DEADLINE:
        added = False
        # Try to add vertex n
        for _ in range(max_attempts_per_vertex):
            if time.time() - start_time > DEADLINE:
                break
            
            # Generate random mask for new vertex
            # We can bias towards certain degrees, but random is fine for start
            # Let's try random masks.
            # Note: The new vertex is index n.
            # We need to check against existing K4s and I4s.
            
            # Random mask
            mask = 0
            for i in range(n):
                if random.random() < 0.5:
                    mask |= (1 << i)
            
            # Check validity
            valid = True
            
            # Check K5: Any K4 C where C is subset of mask?
            # C is a bitmask of 4 vertices.
            # C subset mask iff (C & mask) == C
            for c in k4_list:
                if (c & mask) == c:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check I5: Any I4 I where I is subset of ~mask?
            # ~mask is infinite bits, but we only care about bits 0..n-1.
            # So we check if I is subset of (full_mask ^ mask)
            full_mask = (1 << n) - 1
            non_neigh_mask = full_mask ^ mask
            for i4 in i4_list:
                if (i4 & non_neigh_mask) == i4:
                    valid = False
                    break
            
            if valid:
                # Accept new vertex
                adj.append(mask)
                # Update adjacency for existing vertices
                for i in range(n):
                    if mask & (1 << i):
                        adj[i] |= (1 << n)
                
                # Update lists of K4 and I4
                # New K4s must include the new vertex n.
                # A new K4 containing n is formed by n + a K3 in the neighborhood of n.
                # Similarly, new I4 containing n is formed by n + an I3 in the non-neighborhood of n.
                
                # We need to find all K3s in the subgraph induced by neighbors of n.
                # And all I3s in the subgraph induced by non-neighbors of n.
                
                # To avoid O(n^3) every time, we can just recompute if n is small, 
                # or update incrementally. 
                # Recomputing find_4_cliques() is O(n^4). For n=40, that's 2.5M ops. 
                # With 1000 attempts, this might be too slow if we recompute on every success.
                # But we only recompute on success. Successes should be rare.
                
                # Let's optimize the update:
                # Find all K3s in the neighborhood.
                # Neighbors are bits set in mask.
                neigh_list = []
                for i in range(n):
                    if mask & (1 << i):
                        neigh_list.append(i)
                        
                new_k4s = []
                for a_idx in range(len(neigh_list)):
                    a = neigh_list[a_idx]
                    for b_idx in range(a_idx+1, len(neigh_list)):
                        b = neigh_list[b_idx]
                        if not (adj[a] & (1 << b)): continue
                        for c_idx in range(b_idx+1, len(neigh_list)):
                            c = neigh_list[c_idx]
                            if not (adj[a] & (1 << c)): continue
                            if not (adj[b] & (1 << c)): continue
                            # Found K3 {a,b,c} in neighborhood
                            # New K4 is {a,b,c,n}
                            new_k4s.append((1 << a) | (1 << b) | (1 << c) | (1 << n))
                
                # Find all I3s in the non-neighborhood.
                non_neigh_list = []
                for i in range(n):
                    if not (mask & (1 << i)):
                        non_neigh_list.append(i)
                        
                new_i4s = []
                for a_idx in range(len(non_neigh_list)):
                    a = non_neigh_list[a_idx]
                    for b_idx in range(a_idx+1, len(non_neigh_list)):
                        b = non_neigh_list[b_idx]
                        if adj[a] & (1 << b): continue
                        for c_idx in range(b_idx+1, len(non_neigh_list)):
                            c = non_neigh_list[c_idx]
                            if adj[a] & (1 << c): continue
                            if adj[b] & (1 << c): continue
                            # Found I3 {a,b,c} in non-neighborhood
                            # New I4 is {a,b,c,n}
                            new_i4s.append((1 << a) | (1 << b) | (1 << c) | (1 << n))
                            
                k4_list.extend(new_k4s)
                i4_list.extend(new_i4s)
                
                n += 1
                added = True
                best_n = n
                best_edges = get_edges()
                break # Move to next vertex
                
        if not added:
            # Could not add vertex n with random attempts
            # We could try a different strategy or break
            # Breaking is safer to return a valid graph
            break

    # Return the best graph found
    # Ensure we return the edges for the best_n found
    # If we broke early, best_edges corresponds to best_n.
    # If we finished the loop, best_edges corresponds to current n.
    
    return best_n, best_edges
