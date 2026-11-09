def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def chromatic_number_and_coloring(adj, n):
        """
        Find chromatic number and coloring of graph 'adj' with 'n' vertices.
        adj: list of sets, adj[i] contains neighbors of i.
        Returns (chi, colors) where colors is a list of lists of vertices.
        """
        if n == 0:
            return 0, []
        
        # Order vertices by degree descending (largest degree first) to help backtracking
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        # Map original vertex to position in order? 
        # Or just use the order for assignment.
        
        best_chi = n
        best_coloring = None
        
        # We can use a branch and bound approach.
        # Try to color with k colors.
        
        def try_color(k_colors):
            # colors[i] = color assigned to vertex i (0-indexed in order)
            # Actually, let's assign colors to vertices in 'order'.
            # color_of_vertex[i] = c
            color_of_vertex = [-1] * n
            
            # Pre-compute available colors?
            # Backtracking
            def backtrack(idx):
                nonlocal best_chi, best_coloring
                if idx == n:
                    # Check if this is better
                    if k_colors < best_chi:
                        best_chi = k_colors
                        # Extract coloring
                        cols = [[] for _ in range(k_colors)]
                        for v in range(n):
                            cols[color_of_vertex[v]].append(v)
                        best_coloring = cols
                    return True
                
                v = order[idx]
                
                # Determine which colors are forbidden
                forbidden = set()
                for u in adj[v]:
                    if color_of_vertex[u] != -1:
                        forbidden.add(color_of_vertex[u])
                
                # Try colors
                # Optimization: try colors that appear most often? Or just 0..k_colors-1
                # We can also prune: if number of used colors + (n - idx) < best_chi? No.
                
                # Heuristic: try colors that are already used first?
                # Or try new color last?
                
                for c in range(k_colors):
                    if c in forbidden:
                        continue
                    
                    color_of_vertex[v] = c
                    if backtrack(idx + 1):
                        return True
                    color_of_vertex[v] = -1
                
                return False
            
            return backtrack(0)

        # Binary search or linear search for min k?
        # Linear search from lower bound.
        # Lower bound: max degree + 1, or clique number.
        # Let's just iterate k from 1 to n.
        for k in range(1, n + 1):
            if try_color(k):
                return best_chi, best_coloring
            # If try_color fails, best_chi might have been updated? 
            # No, try_color only updates if it finds a valid coloring.
            # So if it returns False, no coloring with k colors.
            
        return best_chi, best_coloring

    def split_graph_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        C = list(range(k))
        I = list(range(k, n))
        
        # Build adjacency for I-C edges
        # neighbors_in_C[i] for i in I
        neighbors_in_C = {i: set() for i in I}
        for u, v in edge_set:
            if u < k and v >= k:
                neighbors_in_C[v].add(u)
            elif v < k and u >= k:
                neighbors_in_C[u].add(v)
        
        cliques = []
        
        # 1. Form cliques for I vertices
        for i in I:
            N_i = neighbors_in_C[i]
            if N_i:
                clique = [i] + sorted(N_i)
                cliques.append(clique)
            else:
                # If no neighbors, i is isolated? 
                # But split graph definition: cross pair joined with prob p.
                # If p < 1, possible to have no neighbors.
                # But task says "Every listed set must be a clique with at least two vertices".
                # If i has no neighbors, it has no edges. So it doesn't need to be in any clique?
                # "every edge must lie in exactly one listed clique".
                # If i has no edges, it's fine.
                pass

        # 2. Construct H on C
        # H has edge (c, c') if exists i in I such that c, c' in N_i
        # adj_H[c] = set of c' such that (c, c') in H
        adj_H = [set() for _ in range(k)]
        
        for i in I:
            N_i = neighbors_in_C[i]
            # All pairs in N_i are in H
            N_list = list(N_i)
            for idx, c1 in enumerate(N_list):
                for c2 in N_list[idx+1:]:
                    adj_H[c1].add(c2)
                    adj_H[c2].add(c1)
        
        # 3. Find coloring of H
        chi, coloring = chromatic_number_and_coloring(adj_H, k)
        
        # 4. Add cliques from coloring
        # Each color class is an independent set in H, which is a clique in U (and C)
        for color_class in coloring:
            if len(color_class) >= 2:
                # Sort for consistency?
                cliques.append(sorted(color_class))
            # If len < 2, skip (no edges to cover)
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = split_graph_partition(n, k, p, seed)
        results.append(cliques)
        
    return results
