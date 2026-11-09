"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from itertools import combinations
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def greedy_clique_partition(edge_list, n):
        """Find a clique partition using greedy maximal clique extraction."""
        # Build adjacency from edges
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
        
        if not edge_set:
            return []
        
        cliques = []
        remaining = set(edge_set)
        
        # Precompute: for each vertex, its neighbors
        # Greedy: repeatedly find a large clique in remaining edges
        while remaining:
            # Find a good starting vertex: one with high degree in remaining graph
            deg = [0] * n
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Pick vertex with max degree
            start = max(range(n), key=lambda i: deg[i])
            
            # Build a candidate clique by greedily adding vertices
            # Start with start, then add vertices that are connected to all current clique members
            clique = [start]
            candidates = set(adj[start]) & set(start for s, _ in remaining if s == start) | set(e for e, _ in remaining if e == start) | set(s for s, e in remaining if e == start)
            # Simpler: candidates are neighbors of start that have edges to start in remaining
            candidates = set()
            for u, v in remaining:
                if u == start:
                    candidates.add(v)
                elif v == start:
                    candidates.add(u)
            
            # Greedily build clique: try to add vertices with high connectivity
            while True:
                # Find candidate that is connected to all current clique members in remaining graph
                best = None
                best_score = -1
                for c in candidates:
                    if c in clique:
                        continue
                    # Check if c is connected to all in clique
                    connected = True
                    for m in clique:
                        if (min(c,m), max(c,m)) not in remaining:
                            connected = False
                            break
                    if connected:
                        # Score: degree in remaining graph
                        d = deg[c]
                        if d > best_score:
                            best_score = d
                            best = c
                if best is None:
                    break
                clique.append(best)
                candidates.discard(best)
            
            # Only use clique if size >= 2
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                # Remove all edges within this clique
                to_remove = set()
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, v = clique[i], clique[j]
                        to_remove.add((min(u,v), max(u,v)))
                remaining -= to_remove
            else:
                # Fallback: just take one edge
                u, v = next(iter(remaining))
                cliques.append([u, v])
                remaining.discard((min(u,v), max(u,v)))
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = greedy_clique_partition(edges, n)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
