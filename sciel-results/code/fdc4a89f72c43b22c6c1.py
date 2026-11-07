def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def get_maximal_clique(start, edges):
        """
        Find a maximal clique in the graph defined by 'edges' starting from 'start'.
        Uses a greedy approach: repeatedly add a neighbor connected to all current members.
        To make it deterministic and potentially better, we sort candidates by degree in the candidate set.
        """
        if start not in edges and not any(v == start for u, v in edges):
             # Check if start is isolated?
             pass
             
        clique = [start]
        # Find all neighbors of start
        neighbors = set()
        for u, v in edges:
            if u == start:
                neighbors.add(v)
            elif v == start:
                neighbors.add(u)
        
        # We want to add vertices from neighbors that are connected to all in clique.
        # We iterate until no more can be added.
        # To be efficient, we maintain a set of candidates.
        candidates = set(neighbors)
        
        while True:
            # Filter candidates: must be connected to all in clique
            valid_candidates = []
            for c in candidates:
                is_connected_to_all = True
                for m in clique:
                    if m == c: continue
                    e = (m, c) if m < c else (c, m)
                    if e not in edges:
                        is_connected_to_all = False
                        break
                if is_connected_to_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                break
            
            # Pick the one with highest degree in the remaining candidate set?
            # Or just pick the first one?
            # Let's pick the one with highest degree in the subgraph induced by valid_candidates + clique?
            # Simpler: just pick the one with highest degree in the original remaining graph?
            # Let's sort by degree in the full remaining graph to prioritize "central" nodes.
            def deg(v):
                d = 0
                for u, w in edges:
                    if u == v or w == v:
                        d += 1
                return d
            
            # Optimization: Pre-compute degrees?
            # For now, just pick the max degree one.
            best_c = max(valid_candidates, key=lambda x: deg(x))
            
            clique.append(best_c)
            # Remove best_c from candidates
            candidates.remove(best_c)
            
        return clique

    def count_edges_in_clique(clique, edges):
        count = 0
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                e = (u, v) if u < v else (v, u)
                if e in edges:
                    count += 1
        return count

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        remaining_edges = set(edges)
        
        cliques = []
        
        while remaining_edges:
            # Calculate degrees to find high-degree vertices
            degrees = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                degrees[u] += 1
                degrees[v] += 1
            
            # Get top K vertices by degree
            # K=5 is a good tradeoff
            top_verts = sorted([v for v in range(n) if degrees[v] > 0], key=lambda x: -degrees[x])[:5]
            
            best_clique = []
            best_edge_count = -1
            
            for v in top_verts:
                clique = get_maximal_clique(v, remaining_edges)
                if len(clique) < 2:
                    continue
                # Ensure it's a clique in remaining_edges (safety)
                is_clique = True
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        e = (u, w) if u < w else (w, u)
                        if e not in remaining_edges:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if not is_clique:
                    continue
                
                edge_count = count_edges_in_clique(clique, remaining_edges)
                if edge_count > best_edge_count:
                    best_edge_count = edge_count
                    best_clique = clique
            
            if not best_clique:
                # Fallback: pick any edge
                u, v = next(iter(remaining_edges))
                best_clique = [u, v]
                best_edge_count = 1
            
            cliques.append(best_clique)
            
            # Remove edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, w = best_clique[i], best_clique[j]
                    e = (u, w) if u < w else (w, u)
                    remaining_edges.discard(e)
            
        results.append(cliques)
    
    return results
