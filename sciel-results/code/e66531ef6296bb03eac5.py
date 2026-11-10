"""Clique partition of split graphs using structural properties."""
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
        # Build cross-adjacency
        cross_adj_C = [set() for _ in range(k)]
        
        for u, v in edges:
            if u < k and v >= k:
                cross_adj_C[u].add(v)
            elif u >= k and v < k:
                cross_adj_C[v].add(u)
        
        # Partition C into disjoint groups greedily
        # Each group G gets a clique G ∪ N(G) where N(G) = common neighbors in I
        # This covers: all C-C edges within G, and all cross-edges from G to N(G)
        # Remaining cross-edges (from G to I \ N(G)) become 2-cliques
        
        cliques = []
        unassigned = list(range(k))
        
        while unassigned and time.time() < deadline:
            # Pick the vertex with largest neighborhood to start a group
            best_idx = max(range(len(unassigned)), key=lambda i: len(cross_adj_C[unassigned[i]]))
            start = unassigned[best_idx]
            unassigned.pop(best_idx)
            
            group = [start]
            start_neighbors = set(cross_adj_C[start])
            
            # Add other unassigned vertices that share ALL of start's neighbors
            # (so the common neighborhood is still large)
            new_unassigned = []
            for u in unassigned:
                if start_neighbors.issubset(cross_adj_C[u]):
                    group.append(u)
                else:
                    new_unassigned.append(u)
            unassigned = new_unassigned
            
            # Compute common neighbors of the entire group
            common_I = set(cross_adj_C[group[0]])
            for u in group[1:]:
                common_I &= set(cross_adj_C[u])
            
            # Form clique: group + common_I
            clique_vertices = group + list(common_I)
            if len(clique_vertices) >= 2:
                cliques.append(clique_vertices)
            
            # Now, for each u in group, the cross-edges to I \ common_I need coverage
            # These become 2-cliques
            for u in group:
                remaining_I = cross_adj_C[u] - common_I
                for v in remaining_I:
                    cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
        
        if time.time() > deadline:
            break
    
    return results
# EVOLVE-BLOCK-END
