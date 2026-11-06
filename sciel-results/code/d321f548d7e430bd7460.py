import time

def construct():
    start_time = time.time()
    timeout = 85.0
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # adj[i] is a set of nodes j compatible with i
    adj = [set() for _ in range(N + 1)]
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if (i * j) % (i + j) != 0:
                adj[i].add(j)
                
    best_clique = []
    
    def solve(P, R):
        nonlocal best_clique
        nonlocal start_time
        
        if time.time() - start_time > timeout:
            return
            
        if not P:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        # Greedy Coloring
        # Calculate degrees in P
        deg = {}
        for u in P:
            deg[u] = 0
        for u in P:
            for v in adj[u]:
                if v in deg:
                    deg[u] += 1
                    
        P_sorted = sorted(P, key=lambda x: deg[x], reverse=True)
        
        colors = [0] * len(P_sorted)
        for i in range(len(P_sorted)):
            vi = P_sorted[i]
            max_c = 0
            for j in range(i):
                vj = P_sorted[j]
                if vj in adj[vi]:
                    if colors[j] > max_c:
                        max_c = colors[j]
            colors[i] = max_c + 1
            
        # Process in reverse order of index in P_sorted (which corresponds to decreasing color roughly? No, color is not strictly monotonic with index, but we sorted by degree).
        # Actually, the color assignment depends on the order.
        # We want to process nodes with higher colors first.
        # Let's sort the indices by color descending.
        
        indices = list(range(len(P_sorted)))
        indices.sort(key=lambda x: colors[x], reverse=True)
        
        for idx in indices:
            node = P_sorted[idx]
            c = colors[idx]
            
            if len(R) + c < len(best_clique):
                # Since colors are not necessarily sorted by index, we can't just break?
                # Wait. The bound `len(R) + c` is valid for this specific node.
                # If this node's bound is too low, can we skip it? Yes.
                # Can we break the loop?
                # Only if we know that all subsequent nodes will have even lower bounds.
                # If we sorted `indices` by color descending, then `c` is non-increasing.
                # So if `len(R) + c < best`, then for all subsequent `idx`, `c' <= c`, so `len(R) + c' <= len(R) + c < best`.
                # So we can break.
                break
                
            # New P: nodes in P_sorted that are "before" idx in the original sorted list?
            # No, the standard algorithm removes the node and its predecessors?
            # The standard algorithm maintains an ordered list.
            # When we pick node at position i, the new candidate set is the intersection of the prefix P[0..i-1] with adj[node].
            # This ensures we don't pick the same node again and maintain the order.
            
            # However, I sorted `P_sorted` by degree.
            # The "prefix" logic relies on the specific ordering used in the coloring step.
            # If I pick a node `node` from `P_sorted`, the new P should be the set of nodes in `P_sorted` that were considered "available" before `node` in the recursive structure.
            # In the standard Tomita, the list is fixed. We iterate i from N to 1.
            # The candidate set for node i is {v_j | j < i and v_j adjacent to v_i}.
            
            # So I must iterate `i` from `len(P_sorted)-1` down to `0`.
            # And I must use the colors assigned in that specific order.
            
            # My previous loop `for idx in indices` breaks the prefix property.
            # I must iterate `i` in descending order of index in `P_sorted`.
            
            pass # Re-implement below

    def solve_correct(P, R):
        nonlocal best_clique
        nonlocal start_time
        
        if time.time() - start_time > timeout:
            return
            
        if not P:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        deg = {}
        for u in P:
            deg[u] = 0
        for u in P:
            for v in adj[u]:
                if v in deg:
                    deg[u] += 1
                    
        P_sorted = sorted(P, key=lambda x: deg[x], reverse=True)
        
        colors = [0] * len(P_sorted)
        for i in range(len(P_sorted)):
            vi = P_sorted[i]
            max_c = 0
            for j in range(i):
                vj = P_sorted[j]
                if vj in adj[vi]:
                    if colors[j] > max_c:
                        max_c = colors[j]
            colors[i] = max_c + 1
            
        # Iterate i from len-1 down to 0
        for i in range(len(P_sorted) - 1, -1, -1):
            node = P_sorted[i]
            c = colors[i]
            
            if len(R) + c < len(best_clique):
                break
                
            new_P = []
            for j in range(i):
                candidate = P_sorted[j]
                if candidate in adj[node]:
                    new_P.append(candidate)
                    
            new_R = R + [node]
            solve_correct(new_P, new_R)

    initial_P = sorted(nodes, key=lambda x: len(adj[x]), reverse=True)
    solve_correct(initial_P, [])
    
    return best_clique
