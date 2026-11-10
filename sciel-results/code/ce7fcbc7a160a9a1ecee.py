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
        # Split graph: clique part C = {0..k-1}, independent set I = {k..n-1}
        # Edges: all within C, some between C and I (no edges within I)
        
        # Build adjacency for cross edges: for each independent vertex, its neighbors in C
        cross_adj = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                cross_adj[v].add(u)
            elif v < k and u >= k:
                cross_adj[u].add(v)
        
        # Strategy: 
        # 1. One clique for all of C (covers all internal C edges)
        # 2. For cross edges, we need to cover each edge (c, i) where c in C, i in I
        #    We can group edges by independent vertex: for vertex i, its neighbors in C form a set N(i)
        #    We can create cliques that include i and subsets of N(i)
        #    But each clique must be a clique: {i} ∪ S where S ⊆ N(i) is a clique in C (always true since C is a clique)
        #    So {i} ∪ S is a clique for any S ⊆ N(i)
        #    
        #    To minimize number of cliques, we want to cover all edges (c,i) for each i with fewest cliques.
        #    Each clique {i} ∪ S covers edges (c,i) for c in S.
        #    We need to cover all c in N(i) with subsets, minimizing total cliques across all i.
        #    
        #    Greedy: for each i, if |N(i)| >= 2, we can use one clique {i} ∪ N(i) which covers all edges for i.
        #    If |N(i)| == 1, we need one clique {i, c} for that single edge.
        #    If |N(i)| == 0, no edges.
        #    
        #    But wait - can we do better by sharing cliques across different i values?
        #    No, because each clique can only contain one independent vertex (since I is independent).
        #    So cliques are either:
        #    - Subset of C (covers internal C edges)
        #    - {i} ∪ S where i in I, S ⊆ C (covers cross edges)
        #    
        #    Since C is a clique, any subset of C is a clique. So {i} ∪ S is always a clique.
        #    
        #    Optimal for each i: use one clique {i} ∪ N(i) if |N(i)| >= 2, or one clique {i, c} if |N(i)| == 1.
        #    This gives min(|N(i)|, 1 if |N(i)|>=1 else 0) cliques per i... wait no.
        #    If |N(i)| = m, we need to cover m edges. One clique {i} ∪ N(i) covers all m edges. So 1 clique suffices if m >= 1.
        #    
        #    So total cliques = 1 (for C) + sum over i in I of (1 if |N(i)| >= 1 else 0)
        
        cliques = [list(range(k))]
        
        for i in range(k, n):
            neighbors = cross_adj[i]
            if len(neighbors) >= 2:
                cliques.append([i] + list(neighbors))
            elif len(neighbors) == 1:
                c = next(iter(neighbors))
                cliques.append([i, c])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return results
# EVOLVE-BLOCK-END
