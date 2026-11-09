    def compute_suffix_cliques(order, conflict):
        m = len(order)
        suf_clique = [0] * (m + 1)
        # suf_clique[i] is max clique size in subgraph induced by order[i:]
        
        # We need a helper to find max clique in a set of vertices
        def max_clique_in(vertices):
            if not vertices:
                return 0
            # Simple B&B
            max_size = 0
            # Use a list for stack
            # Or just recursion
            def expand(clique, candidates):
                nonlocal max_size
                if not candidates:
                    if len(clique) > max_size:
                        max_size = len(clique)
                    return
                if len(clique) + len(candidates) <= max_size:
                    return
                # Heuristic: pick vertex with max degree in candidates
                # For small sets, just pick first
                v = next(iter(candidates))
                # Include v
                new_cands = candidates & conflict[v]
                expand(clique | {v}, new_cands)
                # Exclude v
                candidates.discard(v)
                expand(clique, candidates)
            
            expand(set(), set(vertices))
            return max_size

        for i in range(m - 1, -1, -1):
            v = order[i]
            # Neighbors of v in the suffix order[i+1:]
            # We need to know which neighbors are in the suffix
            # Since order is fixed, we can precompute for each v, the list of neighbors that appear after v in the order
            # Let's create a map: vertex -> list of neighbors
            # And a set of vertices in the suffix
            
            # For efficiency, precompute `neighbors_in_suffix[i]`?
            # Actually, we can just filter.
            suffix_vertices = set(order[i+1:])
            neigh_in_suffix = [u for u in conflict[v] if u in suffix_vertices]
            
            if not neigh_in_suffix:
                suf_clique[i] = suf_clique[i+1]
            else:
                # Max clique containing v is 1 + max_clique(neigh_in_suffix)
                # But max_clique(neigh_in_suffix) is the max clique in the induced subgraph of neigh_in_suffix
                # Note: max_clique_in expects a set of vertices
                omega_neigh = max_clique_in(set(neigh_in_suffix))
                suf_clique[i] = max(suf_clique[i+1], 1 + omega_neigh)
        return suf_clique
