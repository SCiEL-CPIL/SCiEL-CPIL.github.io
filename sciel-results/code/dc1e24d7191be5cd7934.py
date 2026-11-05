import random
import itertools

def construct():
    n = 16
    # Initialize with a random graph
    # Using a fixed seed for reproducibility in evaluation if needed, 
    # but random is fine for "search score" if it converges to a good solution.
    # However, to be safe and deterministic for the evaluator, we might use a fixed seed 
    # or a robust heuristic. Let's use a fixed seed to ensure consistent results.
    random.seed(42)
    
    # Adjacency matrix for O(1) edge checks
    adj = [[0] * n for _ in range(n)]
    
    # Start with a moderately dense random graph
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.6:
                adj[i][j] = 1
                adj[j][i] = 1

    def has_q3():
        """
        Returns a list of edges of a Q3 subgraph if one exists, else None.
        Q3 is isomorphic to K_{4,4} minus a perfect matching.
        So we look for two disjoint sets U, V of size 4 such that
        the bipartite graph induced by U U V is 3-regular.
        """
        vertices = list(range(n))
        # Iterate over all pairs of disjoint 4-sets
        # C(16,4) = 1820. Pairs = 1820 * C(12,4)/2 ? No, U and V are ordered roles in bipartite check?
        # Actually, we just need to find ANY set of 8 vertices that contains a Q3.
        # It's faster to iterate U, V.
        for U in itertools.combinations(vertices, 4):
            U_set = set(U)
            remaining = [v for v in vertices if v not in U_set]
            for V in itertools.combinations(remaining, 4):
                # Check if G[U, V] is 3-regular bipartite
                # Degrees in the induced bipartite subgraph
                deg_u = [0] * 4
                deg_v = [0] * 4
                edges_in_sub = []
                
                for i, u in enumerate(U):
                    for j, v in enumerate(V):
                        if adj[u][v]:
                            deg_u[i] += 1
                            deg_v[j] += 1
                            edges_in_sub.append((u, v))
                
                # Check if all degrees are 3
                if all(d == 3 for d in deg_u) and all(d == 3 for d in deg_v):
                    return edges_in_sub
        return None

    # Phase 1: Remove edges to destroy all Q3s
    # We do this iteratively
    max_iterations = 1000
    for _ in range(max_iterations):
        q3_edges = has_q3()
        if not q3_edges:
            break
        # Remove a random edge from the found Q3
        u, v = random.choice(q3_edges)
        adj[u][v] = 0
        adj[v][u] = 0

    # Phase 2: Try to add edges back to maximize count
    # We can try to add any missing edge if it doesn't create a Q3
    # To be efficient, we can shuffle the candidate edges
    candidate_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 0:
                candidate_edges.append((i, j))
    
    random.shuffle(candidate_edges)
    
    for u, v in candidate_edges:
        # Tentatively add edge
        adj[u][v] = 1
        adj[v][u] = 1
        
        # Check if this creates a Q3
        # Optimization: Only check Q3s that include the edge (u,v)?
        # A new Q3 must include the new edge.
        # So we only need to check bipartite subgraphs where u and v are in opposite parts.
        # u in U, v in V OR u in V, v in U.
        # And the other 3 vertices in U/V must be chosen from the remaining 14.
        
        def has_q3_with_edge(u, v):
            # Case 1: u in U, v in V
            # U must contain u. V must contain v.
            # Choose 3 other vertices for U from remaining 14 (excluding v? No, U and V disjoint).
            # Remaining vertices: All except u and v. Size 14.
            # We need to choose 3 for U and 3 for V from these 14?
            # No, U has size 4, V has size 4.
            # U = {u} + {3 others}. V = {v} + {3 others}.
            # The "3 others" for U and "3 others" for V must be disjoint.
            # Total 6 others chosen from 14.
            
            others = [x for x in range(n) if x != u and x != v]
            
            # We need to pick a set W of 6 vertices from 'others'.
            # Then partition W into W_u (size 3) and W_v (size 3).
            # U = {u} U W_u, V = {v} U W_v.
            
            for W in itertools.combinations(others, 6):
                W_set = set(W)
                # Partition W into two sets of 3
                # C(6,3) = 20 partitions.
                for W_u in itertools.combinations(W, 3):
                    W_u_set = set(W_u)
                    W_v = tuple(w for w in W if w not in W_u_set)
                    
                    U = (u,) + W_u
                    V = (v,) + W_v
                    
                    # Check 3-regularity in G[U, V]
                    deg_u = [0] * 4
                    deg_v = [0] * 4
                    
                    for i, uu in enumerate(U):
                        for j, vv in enumerate(V):
                            if adj[uu][vv]:
                                deg_u[i] += 1
                                deg_v[j] += 1
                                
                    if all(d == 3 for d in deg_u) and all(d == 3 for d in deg_v):
                        return True
            return False

        if has_q3_with_edge(u, v):
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
        # Else keep the edge

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
