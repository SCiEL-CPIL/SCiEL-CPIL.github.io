"""Strategy: Attempt to find a universal neighbor in I for the core C.
If found, use a clique C U {i} to cover all core edges and all cross edges to i.
Cover remaining cross edges with 2-cliques.
If not found, fall back to the standard optimal strategy (1 big clique + 2-cliques for all cross edges).
"""

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        # Generate the graph structure
        rng = Random(seed)
        
        # Determine cross edges
        # We need to know the neighborhood of each i in I
        # and check if any i is connected to all of C
        
        cross_edges = []
        neighbors = {} # i -> set of u in C
        
        for v in range(k, n):
            neighbors[v] = set()
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].add(u)
                    cross_edges.append((u, v))
        
        # Check for universal neighbor
        universal_i = None
        for i in range(k, n):
            if len(neighbors[i]) == k:
                universal_i = i
                break
        
        if universal_i is not None:
            # Use the optimized strategy
            cliques = []
            # One big clique C U {universal_i}
            cliques.append(list(range(k)) + [universal_i])
            
            # Cover remaining cross edges
            for i in range(k, n):
                if i == universal_i:
                    continue
                for u in neighbors[i]:
                    cliques.append([u, i])
            
            return cliques
        else:
            # Fall back to standard strategy
            cliques = [list(range(k))]
            for i in range(k, n):
                for u in neighbors[i]:
                    cliques.append([u, i])
            return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))

    return results
