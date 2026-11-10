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
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Build adjacency for cross edges
        # N[i] = set of C-vertices adjacent to I-vertex i
        N = {}
        for i in range(k, n):
            N[i] = set()
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        # C-edges
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
            
            # Greedily partition N[i] into groups, avoiding conflicts with
            # already-covered C-edges
            remaining_N = set(N[i])
            
            while remaining_N:
                if time.time() > deadline:
                    break
                
                # Find the largest subset S of remaining_N such that no edge
                # within S is already covered by another cross-clique
                # and S has at least 2 elements (for a valid clique with i)
                # Actually S just needs to be non-empty; clique is {i} ∪ S, size >= 2
                
                # Greedy: build S by adding vertices one by one
                # A vertex u can be added to S if for all v already in S,
                # (min(u,v), max(u,v)) is not in covered_C_edges
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
                    # Shouldn't happen if remaining_N is non-empty
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
        
        # Cover remaining C-edges with C-only cliques using greedy edge-covering
        remaining_C_edges = C_edges - covered_C_edges
        
        if remaining_C_edges:
            # Build adjacency for remaining C-edges
            remaining_adj = defaultdict(set)
            for u, v in remaining_C_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)
            
            while remaining_C_edges:
                if time.time() > deadline:
                    break
                
                # Pick an uncovered edge (u, v)
                u, v = next(iter(remaining_C_edges))
                
                # Find the largest clique containing edge (u,v) in remaining graph
                # Start with {u, v}, then add all common neighbors (they form a clique
                # with u and v since C is a clique)
                S = {u, v}
                common = remaining_adj[u] & remaining_adj[v]
                S |= common
                
                S_list = sorted(S)
                cliques.append(S_list)
                
                # Remove all edges within S from remaining
                for a_idx in range(len(S_list)):
                    for b_idx in range(a_idx + 1, len(S_list)):
                        remaining_C_edges.discard((S_list[a_idx], S_list[b_idx]))
                
                # Update remaining_adj
                for node in S_list:
                    remaining_adj.pop(node, None)
                    for w in list(remaining_adj.keys()):
                        remaining_adj[w].discard(node)
                # Clean up empty entries
                remaining_adj = {u2: v2 for u2, v2 in remaining_adj.items() if v2}
        
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
