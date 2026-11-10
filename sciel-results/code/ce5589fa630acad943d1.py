"""Clique partition of split graphs using structural properties."""
import time
from random import Random
from collections import defaultdict

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
        # 1. Parse edges to find cross edges and build neighborhoods
        # cross_edges: set of (c, i)
        # neighbors_of_i: dict i -> set of c
        neighbors_of_i = {i: set() for i in range(k, n)}
        internal_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((min(u, v), max(u, v)))
            elif (u < k and v >= k) or (u >= k and v < k):
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                neighbors_of_i[i].add(c)
        
        cliques = []
        covered_internal = set()
        
        # 2. Group independent vertices by their neighborhood in K
        # Group key: frozenset of neighbors
        groups = defaultdict(list)
        for i in range(k, n):
            if neighbors_of_i[i]: # Only group those with edges
                key = frozenset(neighbors_of_i[i])
                groups[key].append(i)
        
        # 3. Create cliques from groups
        for neighborhood, vertices_i in groups.items():
            C = list(neighborhood)
            S = vertices_i
            clique_verts = C + S
            if len(clique_verts) >= 2:
                # This clique is valid
                cliques.append(clique_verts)
                # Mark internal edges in C as covered
                for u_idx in range(len(C)):
                    for v_idx in range(u_idx + 1, len(C)):
                        u, v = C[u_idx], C[v_idx]
                        covered_internal.add((min(u, v), max(u, v)))
        
        # 4. Handle remaining internal edges
        for u, v in internal_edges:
            if (u, v) not in covered_internal:
                cliques.append([u, v])
                # No need to mark, as we just added it
        
        # 5. Cleanup types
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
            
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
