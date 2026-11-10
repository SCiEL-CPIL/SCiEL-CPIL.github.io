"""Clique partition of split graphs using neighborhood grouping for cross-edges."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build edge set
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        # Separate edges into clique-internal and cross-edges
        clique_edges = set()
        cross_edges = set()
        for (u, v) in edge_set:
            if u < k and v < k:
                clique_edges.add((u, v))
            else:
                cross_edges.add((u, v))
        
        cliques = []
        
        # Handle clique-internal edges: all clique vertices form a complete subgraph
        # We need to cover all edges among 0..k-1
        # Since it's a complete graph K_k, we can use a single clique if k >= 2
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Handle cross-edges: group independent set vertices by their neighborhood in clique
        # For each independent vertex v (k <= v < n), its neighborhood in clique is
        # the set of clique vertices u (0 <= u < k) such that (u, v) in cross_edges
        
        # Build neighborhood map: for each independent vertex, get its clique neighbors
        neigh_map = {}
        for v in range(k, n):
            neigh = frozenset(u for (u, w) in cross_edges if w == v)
            if neigh:  # Only include vertices that have at least one cross-edge
                neigh_map[v] = neigh
        
        # Group independent vertices by identical neighborhood
        groups = {}
        for v, neigh in neigh_map.items():
            if neigh not in groups:
                groups[neigh] = []
            groups[neigh].append(v)
        
        # For each group with identical neighborhood, create one clique
        # containing the entire clique set (0..k-1) and all independent vertices in the group
        # This covers all cross-edges for those vertices
        for neigh, vertices in groups.items():
            if not vertices:
                continue
            # The clique must include all clique vertices that are neighbors (which is 'neigh')
            # But to form a valid clique, all vertices in the clique must be mutually adjacent
            # Clique vertices 0..k-1 are all mutually adjacent (complete subgraph)
            # Independent vertices have no edges between them, so we can't put multiple independent vertices in one clique
            # Wait - this is a problem. Independent set vertices have no edges between them.
            # So a clique can contain at most ONE independent vertex.
            
            # Therefore, we cannot group multiple independent vertices into one clique.
            # We must create one clique per independent vertex that has cross-edges.
            # Each such clique contains the independent vertex and all its clique neighbors.
            
            for v in vertices:
                clique_vertices = list(neigh) + [v]
                if len(clique_vertices) >= 2:
                    cliques.append(clique_vertices)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
