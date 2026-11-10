def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for spec in specs:
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Initialize neighbor lists for vertices in I (index k to n-1)
        # neighbors[i] will hold the list of neighbors in C (index 0 to k-1)
        neighbors = {i: [] for i in range(k, n)}
        
        # Process edges
        for u, v in edges:
            if u < k and v < k:
                # Internal edge in C, covered by the big clique C
                pass
            elif u < k and v >= k:
                # Cross edge: u in C, v in I
                neighbors[v].append(u)
            elif u >= k and v < k:
                # Cross edge: u in I, v in C
                neighbors[u].append(v)
            else:
                # Both in I, shouldn't happen in split graph generated this way
                pass
        
        # Start with the clique C
        cliques = [list(range(k))]
        
        # Add cliques for each vertex in I that has neighbors
        for i in range(k, n):
            if neighbors[i]:
                # The clique is {i} union its neighbors in C
                # Ensure types are int
                clique = [i] + [int(c) for c in neighbors[i]]
                cliques.append(clique)
                
        results.append(cliques)
        
    return results
