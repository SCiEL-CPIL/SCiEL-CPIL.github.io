"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        edges += cross_edges
        return edges
    
    def solve_split(n, k, p, seed):
        rng = Random(seed)
        # Determine neighbors in C for each vertex in I
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[v] = nbrs
        
        cliques = []
        
        # Cover all edges within C with a single clique C
        # This is valid because C is a clique in the graph.
        cliques.append(list(range(k)))
        
        # Cover cross edges: for each edge (v, u) where v in I, u in C,
        # create a clique {v, u}.
        # Since we already covered all internal C edges with the big clique,
        # we cannot use cross cliques that contain >1 vertex from C,
        # as that would double-cover the internal edge between them.
        for v in range(k, n):
            for u in neighbors[v]:
                cliques.append([v, u])
        
        return cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Return best effort - but we should have finished
            break
        result.append(solve_split(*spec))
    
    return result
# EVOLVE-BLOCK-END
