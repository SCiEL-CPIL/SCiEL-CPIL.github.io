"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Generate internal edges of the clique C
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        # Generate cross edges between C and I
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return edges, cross_edges

    def greedy_clique_cover(edges, vertices):
        """
        Covers a set of edges with cliques.
        edges: list of (u, v)
        vertices: set of vertices involved (subset of C)
        Returns list of cliques (lists of vertices)
        """
        if not edges:
            return []
        
        edge_set = set(edges)
        remaining_edges = set(edges)
        cliques = []
        
        # Map vertex to set of neighbors in remaining graph
        # Since we are working on a subgraph of a clique, we can just check pairs.
        
        while remaining_edges:
            # Pick an arbitrary edge to start a clique
            # Heuristic: pick edge with highest degree sum? Or just first.
            # Given small size, first is fine.
            u, v = next(iter(remaining_edges))
            
            # Find all vertices w such that (u, w) and (v, w) are in remaining_edges
            # w must be distinct from u, v
            candidates = set()
            # We need to check all vertices in the original set 'vertices'
            # Or just iterate through remaining edges to find common neighbors
            
            # Efficient way for small graphs:
            # For each w in vertices:
            #   if (u, w) in remaining and (v, w) in remaining:
            #      candidates.add(w)
            
            for w in vertices:
                if w == u or w == v:
                    continue
                # Check if u-w and v-w are in remaining
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in remaining_edges and e2 in remaining_edges:
                    candidates.add(w)
            
            clique = [u, v] + list(candidates)
            cliques.append(clique)
            
            # Remove all edges covered by this clique
            # Edges are all pairs in clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    e = (a, b) if a < b else (b, a)
                    remaining_edges.discard(e)
                    
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for n, k, p, seed in specs:
        internal_edges, cross_edges = instance(n, k, p, seed)
        
        # 1. Determine N_C(v) for each v in I
        # N_C[v] is set of neighbors in C
        N_C = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            N_C[v].add(u)
            
        # 2. For each v in I, determine E_v (internal edges covered by K_v)
        # and weight w_v = |N_C(v)| - 1
        # We only consider bundling if |N_C(v)| >= 2
        candidates = []
        for v in range(k, n):
            neigh = N_C[v]
            if len(neigh) >= 2:
                # Generate all pairs in neigh
                edges_v = set()
                neigh_list = list(neigh)
                for i in range(len(neigh_list)):
                    for j in range(i + 1, len(neigh_list)):
                        a, b = neigh_list[i], neigh_list[j]
                        e = (a, b) if a < b else (b, a)
                        edges_v.add(e)
                weight = len(neigh) - 1
                candidates.append((v, edges_v, weight))
                
        # Sort by weight descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # 3. Greedy Set Packing
        selected = []
        covered_internal = set()
        
        for v, edges_v, weight in candidates:
            # Check if edges_v are disjoint from covered_internal
            if not (edges_v & covered_internal):
                selected.append(v)
                covered_internal.update(edges_v)
                
        # 4. Construct Cliques
        cliques = []
        
        # Add bundled cliques
        for v in selected:
            neigh = N_C[v]
            clique = list(neigh) + [v]
            cliques.append(clique)
            
        # Add unbundled cross edges as pairs
        for v in range(k, n):
            if v in selected:
                continue
            neigh = N_C[v]
            for u in neigh:
                cliques.append([u, v])
                
        # 5. Cover remaining internal edges
        remaining_internal = set()
        all_internal = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_internal.add((u, v))
                
        # remaining_internal = all_internal - covered_internal
        for e in all_internal:
            if e not in covered_internal:
                remaining_internal.add(e)
                
        if remaining_internal:
            # Get vertices involved in remaining_internal
            verts_involved = set()
            for u, v in remaining_internal:
                verts_involved.add(u)
                verts_involved.add(v)
                
            rem_cliques = greedy_clique_cover(list(remaining_internal), verts_involved)
            cliques.extend(rem_cliques)
            
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
