"""Clique partitions of three fixed split (chordal) graphs."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        """
        Partition the edge set of a split graph into cliques.
        
        Structure:
        - Clique part: vertices 0..k-1 (all edges present among them)
        - Independent set: vertices k..n-1 (no edges among them)
        - Cross edges: between clique part and independent set, present with prob p
        
        Strategy:
        1. One clique for the entire clique part {0, 1, ..., k-1} covers all internal edges.
        2. For cross edges, group them by the independent-set vertex.
           For each independent vertex v (k <= v < n), collect all clique-part vertices
           connected to v. If there are >= 2 such vertices, form a clique {u1, u2, ..., um, v}
           where {u1, ..., um} are the neighbors of v in the clique part. This is a valid
           clique because all ui are in the clique part (mutually adjacent) and v is adjacent
           to each ui.
           
        This gives exactly one clique per independent vertex that has at least 2 neighbors
        in the clique part, plus one clique for the clique part itself.
        """
        cliques = []
        
        # Clique for the clique part
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Group cross edges by independent vertex
        cross_neighbors = {v: [] for v in range(k, n)}
        for (u, v) in edges:
            if u < k and v >= k:
                cross_neighbors[v].append(u)
            elif u >= k and v < k:
                cross_neighbors[u].append(v)
        
        # For each independent vertex with >= 2 neighbors, form a clique
        for v in range(k, n):
            neighbors = cross_neighbors[v]
            if len(neighbors) >= 2:
                clique = sorted(neighbors) + [v]
                cliques.append(clique)
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        n, k, p, seed = spec
        if time.time() > deadline:
            # Should not happen for these small instances, but guard anyway
            break
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
