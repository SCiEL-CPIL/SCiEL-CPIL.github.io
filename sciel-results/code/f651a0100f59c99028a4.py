import numpy as np

def construct():
    n = 31
    target_edges = 91
    
    def count_4_cycles(A):
        # A is n x n adjacency matrix
        # Common neighbors for all pairs
        # C4 count = 0.5 * sum_{u<v} C(common(u,v), 2)
        common = A @ A
        # common[u,v] is number of common neighbors
        # We need sum of C(common[u,v], 2) for u < v
        # C(k, 2) = k*(k-1)/2
        # Only consider upper triangle
        iu = np.triu_indices(n, k=1)
        vals = common[iu]
        return int(np.sum(vals * (vals - 1) / 2))

    def get_edge_4_cycle_counts(A):
        # Returns a dict or array of how many 4-cycles each edge is in
        # Edge (u,v) is in a 4-cycle if there is a path u-w-x-v
        # w in N(u), x in N(v), w!=v, x!=u, and A[w,x]=1
        counts = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    # Find w in N(u) \ {v}
                    # Find x in N(v) \ {u}
                    # Check A[w,x]
                    # Optimization: 
                    # Sum over w in N(u), w!=v of (A[w, :] & A[v, :])
                    # But we need w-x edge.
                    # count = sum_{w in N(u), w!=v} sum_{x in N(v), x!=u} A[w,x]
                    # This is (A[u, :] * A) [v, :]? No.
                    # Let M = A.
                    # Path u-w-x-v.
                    # Sum over w: A[u,w] * (A[w, :] * A[:, v])
                    # But exclude w=v and x=u.
                    # Since A is symmetric and no self-loops:
                    # If w=v, A[u,v]=1, then term is A[v, :] * A[:, v] = deg(v). But w=v is not allowed.
                    # If x=u, A[v,u]=1, term is A[w, u] * 1. But x=u is not allowed.
                    # Actually, the formula sum_{w,x} A[u,w]A[w,x]A[x,v] counts all paths of length 3.
                    # A 4-cycle u-w-x-v-u corresponds to 2 such paths (u-w-x-v and u-v-x-w).
                    # Wait, a 4-cycle is a subgraph.
                    # The number of 4-cycles containing edge (u,v) is:
                    # (1/2) * (number of paths of length 3 from u to v not using edge (u,v) directly? No)
                    # Standard formula:
                    # Number of 4-cycles through (u,v) = sum_{w in N(u), x in N(v), w!=v, x!=u} A[w,x]
                    # Let's compute this.
                    # N_u = np.where(A[u] == 1)[0]
                    # N_v = np.where(A[v] == 1)[0]
                    # valid_w = [w for w in N_u if w != v]
                    # valid_x = [x for x in N_v if x != u]
                    # count = sum(A[w, x] for w in valid_w for x in valid_x)
                    # This is O(deg^2).
                    pass
        # Too slow to implement complex logic in thought, will use numpy vectorization in code.
        # For n=31, even O(n^3) is fine.
        counts = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    # Get neighbors
                    N_u = np.where(A[u] == 1)[0]
                    N_v = np.where(A[v] == 1)[0]
                    # Filter
                    N_u = N_u[N_u != v]
                    N_v = N_v[N_v != u]
                    # Count edges between N_u and N_v
                    # Submatrix A[N_u, N_v]
                    if len(N_u) > 0 and len(N_v) > 0:
                        sub = A[np.ix_(N_u, N_v)]
                        cnt = int(np.sum(sub))
                        counts[u, v] = cnt
                        counts[v, u] = cnt
        return counts

    def remove_edge_most_4_cycles(A):
        counts = get_edge_4_cycle_counts(A)
        # Find max
        max_val = np.max(counts)
        if max_val == 0:
            return None
        # Get index
        idx = np.unravel_index(np.argmax(counts), counts.shape)
        u, v = idx
        A[u, v] = 0
        A[v, u] = 0
        return (u, v)

    def add_edge_min_4_cycles(A, target_increase=1):
        # Find non-edge (u,v) that adds min 4-cycles
        best_u, best_v = -1, -1
        min_cyc = float('inf')
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    # Common neighbors
                    common = int(np.dot(A[u], A[v]))
                    # New 4-cycles = C(common, 2)
                    new_cyc = common * (common - 1) // 2
                    if new_cyc < min_cyc:
                        min_cyc = new_cyc
                        best_u, best_v = u, v
                        if min_cyc == 0:
                            # Can't get better than 0
                            # But we might want to check if there are others? 
                            # No, 0 is min.
                            pass
        if best_u == -1:
            return None
        A[best_u, best_v] = 1
        A[best_v, best_u] = 1
        return (best_u, best_v)

    best_graph = None
    best_score = float('inf')
    
    for seed in range(10):
        np.random.seed(seed)
        # Start with random graph
        p = 0.25
        A = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < p:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Phase 1: Remove edges to eliminate 4-cycles
        # Limit iterations to prevent infinite loops
        for _ in range(100):
            c4 = count_4_cycles(A)
            if c4 == 0:
                break
            remove_edge_most_4_cycles(A)
        
        # Phase 2: Add edges to reach 90
        edges = int(np.sum(A) / 2)
        while edges < 90:
            # Try to add edge with 0 new 4-cycles
            # We need a function to check if adding (u,v) creates 0 4-cycles
            # which is equivalent to common(u,v) <= 1
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        common = int(np.dot(A[u], A[v]))
                        if common <= 1:
                            A[u, v] = 1
                            A[v, u] = 1
                            edges += 1
                            found = True
                            break
                if found:
                    break
            if not found:
                # No edge adds 0 4-cycles
                # If we are at 90, we are done with C4-free.
                # If we are < 90, we cannot reach 90 C4-free.
                # We must accept 4-cycles to reach 91?
                # The task requires 91 edges.
                # If we can't reach 90 C4-free, we should have started denser?
                # Or we just add the best edge and continue.
                # But the goal is MINIMIZE 4-cycles.
                # If we can't maintain 0, we should have stopped earlier?
                # No, we want 91 edges.
                # If we are stuck at <90 with 0 C4, we must add edges that create C4.
                # We should pick the one that creates the fewest.
                break
        
        # Phase 3: Add edges until 91
        while edges < 91:
            res = add_edge_min_4_cycles(A)
            if res is None:
                break
            edges += 1
        
        final_c4 = count_4_cycles(A)
        if final_c4 < best_score:
            best_score = final_c4
            best_graph = A.copy()
            
        # Early exit if we found 0 or 1?
        if best_score <= 1:
            break

    # Extract edges from best_graph
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_graph[u, v]:
                edges_list.append([u, v])
                
    return edges_list
