"""Rewritten program for clique partition of split graphs."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        """
        Structure-aware clique partition for split graphs.
        
        Key insight: The problem decomposes into:
        1. Cross edges: each edge (u,i) with u in C, i in I must be in exactly one clique
        2. C-edges: each edge (u,v) with u,v in C must be in exactly one clique
        
        A clique containing an I-vertex i is of the form {i} ∪ S where S ⊆ N(i).
        This covers cross edges (i,s) for s in S, and C-edges within S.
        
        Strategy:
        - Process I-vertices in order of decreasing |N(i)|
        - For each i, greedily partition N(i) into groups, where each group S
          must be a clique in the "remaining C-edge graph" (edges not yet covered)
        - This ensures no C-edge is covered by two different cross-cliques
        - After all cross-cliques, cover remaining C-edges with C-only cliques
        """
        edges = instance(n, k, p, seed)
        
        # Build adjacency for cross edges
        N = {i: set() for i in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        # C-edges: all pairs within C
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))
        
        # Track which C-edges are covered by cross-cliques
        covered_C_edges = set()
        cliques = []
        
        # Process I-vertices in order of decreasing |N(i)|
        I_vertices = sorted(range(k, n), key=lambda i: len(N[i]), reverse=True)
        
        for i in I_vertices:
            if time.time() > deadline:
                break
            if not N[i]:
                continue
            
            # Greedily partition N[i] into groups
            remaining_N = set(N[i])
            
            while remaining_N:
                if time.time() > deadline:
                    break
                
                # Find the largest subset S of remaining_N such that no edge
                # within S is already covered by another cross-clique
                S = set()
                for u in sorted(remaining_N):
                    can_add = True
                    for v in S:
                        e = (min(u, v), max(u, v))
                        if e in covered_C_edges:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                
                if not S:
                    S = {next(iter(remaining_N))}
                
                # Create clique {i} ∪ S
                clique = sorted([i] + list(S))
                cliques.append(clique)
                
                # Mark C-edges within S as covered
                S_list = sorted(S)
                for a_idx in range(len(S_list)):
                    for b_idx in range(a_idx + 1, len(S_list)):
                        e = (S_list[a_idx], S_list[b_idx])
                        covered_C_edges.add(e)
                
                remaining_N -= S
        
        # Now cover remaining C-edges with C-only cliques
        remaining_C_edges = C_edges - covered_C_edges
        
        if remaining_C_edges:
            # Greedy: repeatedly find a max clique in the remaining C-edge graph
            remaining_adj = defaultdict(set)
            for u, v in remaining_C_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)
            
            while remaining_C_edges:
                if time.time() > deadline:
                    break
                
                # Get vertex with max remaining degree
                if not remaining_adj:
                    break
                v = max(remaining_adj.keys(), key=lambda x: len(remaining_adj[x]))
                
                # Clique = {v} ∪ remaining_adj[v]
                S = {v} | remaining_adj[v]
                S_list = sorted(S)
                
                if len(S_list) < 2:
                    if remaining_adj[v]:
                        w = next(iter(remaining_adj[v]))
                        S_list = sorted([v, w])
                    else:
                        break
                
                cliques.append(S_list)
                
                # Remove all edges within S from remaining
                for a_idx in range(len(S_list)):
                    for b_idx in range(a_idx + 1, len(S_list)):
                        remaining_C_edges.discard((S_list[a_idx], S_list[b_idx]))
                
                # Update remaining_adj
                for u in S_list:
                    remaining_adj.pop(u, None)
                    for w in remaining_adj:
                        remaining_adj[w].discard(u)
                remaining_adj = {u: v for u, v in remaining_adj.items() if v}
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: one edge per clique
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
