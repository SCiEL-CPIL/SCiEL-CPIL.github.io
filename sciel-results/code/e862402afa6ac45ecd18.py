def construct():
    import time
    from random import Random
    import numpy as np
    
    deadline = time.time() + 140  # 10s safety margin
    
    def generate_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        adj = [[False]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u][v] = True
                    adj[v][u] = True
        return edges, adj
    
    def solve_instance(edges, adj, n):
        # Use bitmasks for fast operations
        # remaining: set of edges still uncovered
        remaining = set(edges)
        if not remaining:
            return []
        
        # Build adjacency as sets for the remaining graph
        # We'll maintain remaining as a set of frozensets or (u,v) with u<v
        
        cliques = []
        
        while remaining and time.time() < deadline:
            # Find a good clique in the remaining edge subgraph
            # Degree in remaining graph
            deg = {}
            for (u, v) in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            # Pick vertex with max degree
            if not deg:
                break
            start = max(deg, key=deg.get)
            
            # Greedily expand clique from start
            clique = [start]
            # Candidates: neighbors of start in remaining graph
            candidates = set()
            for (u, v) in remaining:
                if u == start:
                    candidates.add(v)
                elif v == start:
                    candidates.add(u)
            
            while candidates:
                # Find candidate connected to all in clique
                best = None
                best_score = -1
                for c in candidates:
                    # Check if c connects to all in clique
                    ok = True
                    for m in clique:
                        if m == c:
                            continue
                        pair = (min(m, c), max(m, c))
                        if pair not in remaining:
                            ok = False
                            break
                    if ok:
                        # Score: degree in remaining (heuristic)
                        score = deg.get(c, 0)
                        if score > best_score:
                            best_score = score
                            best = c
                if best is None:
                    break
                clique.append(best)
                # Update candidates: must be connected to best
                new_candidates = set()
                for c in candidates:
                    if c == best:
                        continue
                    pair = (min(c, best), max(c, best))
                    if pair in remaining:
                        new_candidates.add(c)
                candidates = new_candidates
            
            # If clique has at least 2 vertices, add it
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove all edges within clique from remaining
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        remaining.discard(pair)
            else:
                # Shouldn't happen, but safety
                break
        
        # Leftover edges as 2-cliques
        for (u, v) in remaining:
            cliques.append([u, v])
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges, adj = generate_graph(n, p, seed)
        cliques = solve_instance(edges, adj, n)
        results.append(cliques)
    
    return results
