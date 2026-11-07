def construct():
    from random import Random
    import time
    import sys

    start_time = time.time()
    DEADLINE = 115  # seconds

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Core edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return k, edges

    def solve_one(n, k, p, seed):
        k_val, edges = instance(n, k, p, seed)
        remaining_edges = set(edges)
        cliques = []
        
        # Precompute neighbors for speed?
        # adj[u] = set of v such that (u,v) in edges
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Helper to check if a set of vertices forms a clique in remaining_edges
        def is_clique_in_rem(vertices):
            verts = list(vertices)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    if u > v: u, v = v, u
                    if (u, v) not in remaining_edges:
                        return False
            return True

        # Greedy search for best clique
        while remaining_edges:
            best_clique = None
            best_size = 0
            
            # 1. Check pure core cliques
            # Greedy max clique on core
            core_verts = list(range(k_val))
            # Sort by degree in remaining_edges within core?
            # Just try a few heuristics
            # Heuristic 1: Greedy add
            current = set()
            # Order vertices by degree in remaining core graph
            core_deg = {}
            for u in core_verts:
                deg = 0
                for v in core_verts:
                    if u == v: continue
                    e = (min(u,v), max(u,v))
                    if e in remaining_edges: deg += 1
                core_deg[u] = deg
            
            sorted_core = sorted(core_verts, key=lambda x: -core_deg[x])
            for u in sorted_core:
                # Check if u connects to all in current
                ok = True
                for w in current:
                    e = (min(u,w), max(u,w))
                    if e not in remaining_edges:
                        ok = False
                        break
                if ok:
                    current.add(u)
            
            if len(current) > best_size:
                best_size = len(current)
                best_clique = current
                
            # 2. Check mixed cliques
            for v in range(k_val, n):
                # Neighbors of v in remaining edges
                rem_neigh = set()
                for u in adj[v]:
                    if u < k_val:
                        e = (min(u,v), max(u,v))
                        if e in remaining_edges:
                            rem_neigh.add(u)
                
                if not rem_neigh: continue
                
                # Find max clique in rem_neigh using remaining core edges
                # Greedy on rem_neigh
                current_m = set()
                # Sort rem_neigh by degree in remaining core graph
                neigh_deg = {}
                for u in rem_neigh:
                    deg = 0
                    for w in rem_neigh:
                        if u == w: continue
                        e = (min(u,w), max(u,w))
                        if e in remaining_edges: deg += 1
                    neigh_deg[u] = deg
                
                sorted_neigh = sorted(rem_neigh, key=lambda x: -neigh_deg[x])
                for u in sorted_neigh:
                    ok = True
                    for w in current_m:
                        e = (min(u,w), max(u,w))
                        if e not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        current_m.add(u)
                
                candidate = current_m | {v}
                if len(candidate) > best_size:
                    best_size = len(candidate)
                    best_clique = candidate
            
            if best_clique is None:
                # Should not happen if remaining_edges not empty
                break
                
            # Add to result
            cliques.append(list(best_clique))
            
            # Remove edges
            verts = list(best_clique)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    e = (min(u,v), max(u,v))
                    remaining_edges.discard(e)
                    
            if time.time() - start_time > DEADLINE:
                # Fallback: just add remaining edges as pairs?
                # This might be slow, but better than timeout
                # Actually, if we timeout, we return what we have?
                # No, must cover all edges.
                # If we run out of time, we can't guarantee validity.
                # But the prompt says "return the best result found".
                # If invalid, score 0.
                # So we must finish.
                # Greedy is fast.
                pass

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() - start_time > DEADLINE - 5:
            break
        results.append(solve_one(*spec))
    
    while len(results) < 3:
        results.append([])
    
    return results
