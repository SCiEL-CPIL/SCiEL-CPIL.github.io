"""Program: minimize total number of cliques in an edge-exact clique partition of two dense random graphs."""
# EVOLVE-BLOCK-START
def construct():
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(edges):
        if not edges:
            return []
        
        # Pre-compute full adjacency for quick lookup if needed, 
        # but we operate on 'uncovered' set.
        uncovered = set(edges)
        cliques = []
        
        # Helper to get neighbors of a vertex in the uncovered subgraph
        def get_neighbors(v, uncovered_edges):
            neighbors = set()
            for (u, w) in uncovered_edges:
                if u == v:
                    neighbors.add(w)
                elif w == v:
                    neighbors.add(u)
            return neighbors

        while uncovered:
            # Find the vertex that is part of the largest potential clique
            # We iterate through all vertices present in uncovered edges
            vertices_in_graph = set()
            for u, v in uncovered:
                vertices_in_graph.add(u)
                vertices_in_graph.add(v)
            
            best_clique = None
            best_size = 0
            
            for v in vertices_in_graph:
                # Get neighbors of v in the uncovered subgraph
                nbrs = get_neighbors(v, uncovered)
                if not nbrs:
                    continue
                
                # We want to find a clique in the subgraph induced by nbrs.
                # Let's use a greedy heuristic: 
                # 1. Sort neighbors by their degree within the subgraph induced by nbrs (or just degree in uncovered)
                # 2. Start with the neighbor with highest degree.
                # 3. Iteratively add neighbors that are connected to all current members.
                
                # To speed up, we can just try to build a maximal clique starting from v.
                # Candidate set for the clique (excluding v) starts as nbrs.
                # We will pick vertices from nbrs one by one.
                
                # Sort nbrs by degree in the full uncovered graph (proxy for density)
                # Or better, degree within nbrs.
                # Let's compute degree within nbrs for each candidate
                deg_in_nbrs = {}
                for n1 in nbrs:
                    deg_in_nbrs[n1] = 0
                for n1 in nbrs:
                    for n2 in nbrs:
                        if n1 == n2: continue
                        # Check if edge (n1, n2) is in uncovered
                        e = (min(n1, n2), max(n1, n2))
                        if e in uncovered:
                            deg_in_nbrs[n1] += 1
                
                # Sort candidates by degree in nbrs, descending
                candidates = sorted(nbrs, key=lambda x: deg_in_nbrs[x], reverse=True)
                
                current_clique = [v]
                # We need to select a subset of candidates that form a clique
                # and are all connected to v (which they are by definition of nbrs).
                
                # Greedy selection:
                # Pick the candidate with highest degree in the remaining candidate set
                # that is connected to all already picked.
                
                selected = []
                remaining_cands = list(candidates)
                
                while remaining_cands:
                    # Find the best candidate
                    best_cand = None
                    best_score = -1
                    for c in remaining_cands:
                        # Check if c is connected to all in 'selected'
                        ok = True
                        for s in selected:
                            e = (min(c, s), max(c, s))
                            if e not in uncovered:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Score: number of connections to other remaining candidates
                        # that are also connected to all in 'selected'
                        score = 0
                        for d in remaining_cands:
                            if d == c: continue
                            e_cd = (min(c, d), max(c, d))
                            if e_cd in uncovered:
                                # Check if d is connected to all in selected
                                ok_d = True
                                for s in selected:
                                    e_ds = (min(d, s), max(d, s))
                                    if e_ds not in uncovered:
                                        ok_d = False
                                        break
                                if ok_d:
                                    score += 1
                        if score > best_score:
                            best_score = score
                            best_cand = c
                    
                    if best_cand is None:
                        break
                    
                    selected.append(best_cand)
                    remaining_cands.remove(best_cand)
                    
                    # Optimization: remove any candidates in remaining_cands that are not connected to best_cand
                    # or not connected to some selected member
                    new_remaining = []
                    for c in remaining_cands:
                        # Must be connected to best_cand
                        e_bc = (min(c, best_cand), max(c, best_cand))
                        if e_bc not in uncovered:
                            continue
                        # Must be connected to all selected (including best_cand now)
                        ok = True
                        for s in selected:
                            e_cs = (min(c, s), max(c, s))
                            if e_cs not in uncovered:
                                ok = False
                                break
                        if ok:
                            new_remaining.append(c)
                    remaining_cands = new_remaining

                full_clique = [v] + selected
                if len(full_clique) > best_size:
                    best_size = len(full_clique)
                    best_clique = full_clique
            
            if best_clique is None or len(best_clique) < 2:
                # Fallback: take any edge
                e = next(iter(uncovered))
                cliques.append(sorted(e))
                uncovered.discard(e)
            else:
                cliques.append(sorted(best_clique))
                # Remove all edges within best_clique from uncovered
                for i in range(len(best_clique)):
                    a = best_clique[i]
                    for j in range(i + 1, len(best_clique)):
                        b = best_clique[j]
                        e = (min(a, b), max(a, b))
                        uncovered.discard(e)

        return cliques

    result = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = random_graph(n, p, seed)
        cliques = partition_cliques(edges)
        result.append(cliques)
    return result
# EVOLVE-BLOCK-END
