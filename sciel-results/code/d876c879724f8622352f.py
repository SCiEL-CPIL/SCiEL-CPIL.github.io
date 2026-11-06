def construct():
    import sys
    sys.setrecursionlimit(100000)

    def generate_edges(n, p, seed):
        # Reproduce verifier's random_graph edge set using random.Random(seed)
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(n, p, seed):
        edges = generate_edges(n, p, seed)
        if not edges:
            return []

        # Adjacency as bitmasks
        # adj[i] is a bitmask where bit j is set if (i, j) is an edge
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Total edge mask? No, we track uncovered edges as a set of tuples or a list.
        # Using a set of tuples is easier for removal, but checking clique validity requires bit ops.
        # Let's maintain `uncovered` as a set of (u, v) with u < v.
        uncovered = set(edges)
        
        # Precompute edge list for quick iteration? No, we need to remove dynamically.
        
        result = []

        while uncovered:
            # Identify vertices involved in uncovered edges
            cand_vertices = set()
            for u, v in uncovered:
                cand_vertices.add(u)
                cand_vertices.add(v)
                
            # If no vertices, break (shouldn't happen)
            if not cand_vertices:
                break

            # Build subgraph adjacency restricted to uncovered edges among cand_vertices
            # sub_adj[v] is a bitmask of neighbors in the subgraph
            sub_adj = {v: 0 for v in cand_vertices}
            for u, v in uncovered:
                sub_adj[u] |= (1 << v)
                sub_adj[v] |= (1 << u)

            # Find maximal cliques in this subgraph using Bron-Kerbosch with bitmasks
            # We want the clique with the most vertices (which implies most edges)
            best_clique_mask = 0
            best_clique_size = 0

            # Bron-Kerbosch with bitmasks
            # R, P, X are bitmasks
            def bron_kerbosch(R, P, X):
                nonlocal best_clique_mask, best_clique_size
                
                if P == 0 and X == 0:
                    # R is a maximal clique
                    size = bin(R).count('1')
                    if size > best_clique_size:
                        best_clique_size = size
                        best_clique_mask = R
                    return

                # Pivot selection: choose u in P|X maximizing |P & adj[u]|
                PX = P | X
                if PX:
                    # Find pivot u
                    # To speed up, we can just pick the first one or one with high degree.
                    # Standard BK picks u to minimize iterations.
                    # For simplicity and speed in this context, we can just iterate P.
                    # However, pivot is crucial for performance.
                    # Let's find u in PX with max popcount(P & sub_adj[u])
                    # sub_adj is a dict, so we need to check if u is in sub_adj.
                    # Note: P and X are subsets of cand_vertices, so their bits correspond to indices in 0..n-1.
                    # sub_adj is defined for all cand_vertices.
                    
                    best_u = -1
                    best_count = -1
                    temp = PX
                    while temp:
                        # Get lowest set bit
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        count = bin(P & sub_adj[u]).count('1')
                        if count > best_count:
                            best_count = count
                            best_u = u
                        temp -= lsb
                    
                    if best_u != -1:
                        # Candidates are P \ N(u)
                        candidates = P & ~(sub_adj[best_u])
                    else:
                        candidates = P
                else:
                    candidates = P

                # Iterate through candidates
                temp_cand = candidates
                while temp_cand:
                    lsb = temp_cand & -temp_cand
                    v = lsb.bit_length() - 1
                    temp_cand -= lsb

                    # Recurse with R | {v}
                    new_R = R | lsb
                    new_P = P & sub_adj[v]
                    new_X = X & sub_adj[v]
                    
                    bron_kerbosch(new_R, new_P, new_X)
                    
                    # Update P and X for next iteration
                    P &= ~lsb
                    X |= lsb

            # Start BK with R=0, P=all cand_vertices, X=0
            # P is a bitmask of all cand_vertices
            P_mask = 0
            for v in cand_vertices:
                P_mask |= (1 << v)
            
            bron_kerbosch(0, P_mask, 0)

            if best_clique_size < 2:
                # Fallback: take a single edge
                e = next(iter(uncovered))
                # Create mask for this edge
                u, v = e
                best_clique_mask = (1 << u) | (1 << v)
                best_clique_size = 2

            # Extract vertices from best_clique_mask
            clique_vertices = []
            temp_mask = best_clique_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                clique_vertices.append(v)
                temp_mask -= lsb
            
            # Remove edges internal to best_clique from uncovered
            # Generate all pairs in clique_vertices
            vs = clique_vertices
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a, b = vs[i], vs[j]
                    if a > b:
                        a, b = b, a
                    uncovered.discard((a, b))

            result.append(sorted(clique_vertices))

        return result

    return [solve(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]
