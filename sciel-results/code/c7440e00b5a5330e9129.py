import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Initialize with Paley graph on 17 vertices
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def get_edges(adj):
        edges = []
        n = len(adj)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def find_independent_sets(adj, size):
        """Find all independent sets of a given size."""
        n = len(adj)
        sets = []
        
        def backtrack(start, current_set, current_mask):
            if len(current_set) == size:
                sets.append(current_set[:])
                return
            if start >= n:
                return
            # Pruning
            if n - start < size - len(current_set):
                return
            
            for v in range(start, n):
                if adj[v] & current_mask == 0:
                    current_set.append(v)
                    backtrack(v + 1, current_set, current_mask | (1 << v))
                    current_set.pop()
        
        backtrack(0, [], 0)
        return sets

    def find_all_independent_sets(adj):
        """Find all independent sets of any size."""
        n = len(adj)
        sets = []
        
        def backtrack(start, current_set, current_mask):
            sets.append(current_set[:])
            for v in range(start, n):
                if adj[v] & current_mask == 0:
                    current_set.append(v)
                    backtrack(v + 1, current_set, current_mask | (1 << v))
                    current_set.pop()
        
        backtrack(0, [], 0)
        return sets

    # Main extension loop
    while time.time() < deadline:
        # Find all independent sets of size 5
        indep_5 = find_independent_sets(adj, 5)
        
        if not indep_5:
            # No independent sets of size 5.
            # Check current alpha
            all_indep = find_all_independent_sets(adj)
            max_alpha = max(len(s) for s in all_indep)
            
            if max_alpha <= 4:
                N = []
            else:
                # Pick a single vertex to connect to, to reduce alpha potential
                # Actually, any N works for the "hit" condition (vacuously true).
                # But we need alpha(G[V\N]) <= 4.
                # Since G has no indep set of size 5, G[V\N] has none.
                # So alpha(G[V\N]) <= 4.
                # So 1 + alpha(G[V\N]) <= 5.
                # So any N works.
                # To be safe, connect to vertex 0.
                N = [0]
            
            # Add vertex
            new_n = n + 1
            new_adj = adj + [0]
            for u in N:
                new_adj[u] |= (1 << new_n)
                new_adj[new_n] |= (1 << u)
            adj = new_adj
            n = new_n
            continue
        
        # We have indep_5. We need to find an independent set N that hits all of them.
        found = False
        # Try small N first
        for size_n in range(1, 6):
            if found: break
            cand_Ns = find_independent_sets(adj, size_n)
            for N in cand_Ns:
                N_mask = 0
                for v in N:
                    N_mask |= (1 << v)
                
                valid = True
                for I in indep_5:
                    I_mask = 0
                    for v in I:
                        I_mask |= (1 << v)
                    if (N_mask & I_mask) == 0:
                        valid = False
                        break
                
                if valid:
                    new_n = n + 1
                    new_adj = adj + [0]
                    for u in N:
                        new_adj[u] |= (1 << new_n)
                        new_adj[new_n] |= (1 << u)
                    adj = new_adj
                    n = new_n
                    found = True
                    break
        
        if not found:
            break

    edges = get_edges(adj)
    return {'n': n, 'edges': edges}
