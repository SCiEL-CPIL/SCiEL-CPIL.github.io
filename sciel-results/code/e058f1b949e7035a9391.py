import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets for fast lookup
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        cliques = []
        remaining_edges = set(edge_set)
        
        # Step 1: Handle independent set vertices
        # For each v in I, if it has neighbors in K, form clique N_K(v) U {v}
        for v in range(k, n):
            if time.time() > deadline:
                break
            neighbors_in_k = [u for u in range(k) if (u, v) in remaining_edges]
            if neighbors_in_k:
                clique = neighbors_in_k + [v]
                cliques.append(clique)
                # Remove all edges in this clique from remaining_edges
                # Edges are (u, v) for u in neighbors_in_k
                # And (u, w) for u, w in neighbors_in_k
                for u in neighbors_in_k:
                    remaining_edges.discard((u, v))
                for i in range(len(neighbors_in_k)):
                    for j in range(i + 1, len(neighbors_in_k)):
                        u, w = neighbors_in_k[i], neighbors_in_k[j]
                        e = (u, w) if u < w else (w, u)
                        remaining_edges.discard(e)
        
        # Step 2: Handle remaining edges in K
        # These are edges (u, w) with u, w < k
        # Build a graph H on vertices 0..k-1 with edges in remaining_edges
        # We need to cover H with cliques.
        
        # Helper to find a maximum clique in a subgraph defined by a set of edges
        def find_max_clique(vertices, edges):
            if not edges:
                return []
            
            # Simple greedy heuristic for max clique, or exact if small
            # Given k <= 30, we can use a backtracking search for max clique
            # But let's start with a greedy one that is fast and usually good
            # Greedy: pick vertex with max degree, add to clique, remove non-neighbors
            
            # Actually, let's use a simple recursive max clique with pruning
            # State: candidates (set), clique (set)
            # We want to maximize len(clique)
            
            best_clique = []
            
            def bron_kerbosch(candidates, clique, excluded):
                nonlocal best_clique
                if not candidates and not excluded:
                    if len(clique) > len(best_clique):
                        best_clique = list(clique)
                    return
                
                # Pivot
                if excluded:
                    pivot = next(iter(excluded))
                elif candidates:
                    pivot = next(iter(candidates))
                else:
                    return
                
                # Candidates that are not neighbors of pivot
                candidates_to_process = [c for c in candidates if c not in adj.get(pivot, set()) or (min(c, pivot), max(c, pivot)) not in edges]
                
                for v in candidates_to_process:
                    bron_kerbosch(candidates & adj.get(v, set()) & {x for x in candidates if (min(v,x), max(v,x)) in edges}, 
                                   clique | {v}, 
                                   excluded & adj.get(v, set()) & {x for x in excluded if (min(v,x), max(v,x)) in edges})
                    candidates.remove(v)
                    excluded.add(v)
            
            # Initialize
            # vertices is the set of active vertices
            # edges is the set of active edges
            # We restrict adj to active edges
            active_adj = {v: set() for v in vertices}
            for u, w in edges:
                if u in active_adj and w in active_adj:
                    active_adj[u].add(w)
                    active_adj[w].add(u)
            
            # Use Bron-Kerbosch with pivot
            # We need to modify the inner function to use active_adj
            best_clique = []
            def bk(candidates, clique, excluded):
                nonlocal best_clique
                if not candidates and not excluded:
                    if len(clique) > len(best_clique):
                        best_clique = list(clique)
                    return
                
                if excluded:
                    pivot = next(iter(excluded))
                elif candidates:
                    pivot = next(iter(candidates))
                else:
                    return
                
                # Process candidates not connected to pivot
                for v in list(candidates):
                    if (v, pivot) not in edges and (pivot, v) not in edges and v != pivot:
                        # v is not a neighbor of pivot in the current graph
                        new_candidates = candidates & active_adj[v]
                        new_excluded = excluded & active_adj[v]
                        bk(new_candidates, clique | {v}, new_excluded)
                        candidates.remove(v)
                        excluded.add(v)
            
            # Start
            initial_candidates = set(vertices)
            bk(initial_candidates, set(), set())
            
            return best_clique

        while remaining_edges:
            if time.time() > deadline:
                # Fallback: cover remaining edges individually
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Get vertices involved in remaining_edges
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
                
            # Find max clique
            max_clique = find_max_clique(active_vertices, remaining_edges)
            
            if not max_clique:
                # Should not happen if remaining_edges is not empty
                # Fallback: take one edge
                u, v = next(iter(remaining_edges))
                max_clique = [u, v]
            
            cliques.append(max_clique)
            
            # Remove edges of max_clique from remaining_edges
            for i in range(len(max_clique)):
                for j in range(i + 1, len(max_clique)):
                    u, v = max_clique[i], max_clique[j]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split(*spec)
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
