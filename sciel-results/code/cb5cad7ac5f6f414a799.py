def construct():
    import time
    start_time = time.time()
    DEADLINE = 120  # seconds
    
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        # Edges within clique part
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Find cross edges and their structure
        # For each independent vertex v (k to n-1), find its neighbors in C
        neighbors = {v: [] for v in range(k, n)}
        cross_edges = set()
        clique_edges = set()
        for (u, v) in edges:
            if u < k and v < k:
                clique_edges.add((u, v))
            elif u >= k and v >= k:
                # Should not happen (independent set)
                pass
            else:
                # Cross edge
                if u < k:
                    a, b = u, v
                else:
                    a, b = v, u
                cross_edges.add((a, b))
                neighbors[b].append(a)
        
        cliques = []
        # One clique for the entire clique part C
        if k >= 2:
            cliques.append(list(range(k)))
        # Size-2 cliques for each cross edge
        for v in range(k, n):
            for u in neighbors[v]:
                cliques.append([u, v])
        
        return cliques
    
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() - start_time > DEADLINE - 5:
            # Return best effort
            break
        results.append(solve_one(*spec))
    
    # If we ran out of time, fill remaining with trivial solution
    while len(results) < 3:
        spec = specs[len(results)]
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        results.append([[list(e)] for e in sorted(edges)])
    
    return results
