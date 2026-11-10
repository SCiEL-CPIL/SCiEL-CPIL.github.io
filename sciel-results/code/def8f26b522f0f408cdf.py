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
        
        Key properties:
        - C = {0..k-1} is a clique
        - I = {k..n-1} is an independent set
        - Any clique in the graph contains at most one vertex from I
        - A clique is either: subset of C (size>=2), or {i} ∪ S where i∈I, S⊆N(i)
        
        Strategy:
        1. For each i in I, N(i) = neighbors in C. The edges (u,i) for u in N(i)
           must be covered by cliques containing i. Each such clique is {i}∪S where
           S⊆N(i). We want to partition N(i) into as few subsets as possible,
           but each subset S also "uses up" the edges within S (since S is a clique).
        
        2. Edges within C not covered by cross-cliques need separate cliques.
        
        Better approach: Think of it as we're assigning each vertex u in C to be
        "shared" among some I-vertices. The clique {i}∪S covers all edges among S
        AND all edges from i to S.
        
        Optimal structure: We can think of partitioning C into groups G_1, ..., G_m.
        For each group G_j, we can attach any subset of I-vertices that are all
        adjacent to all of G_j. But since each I-vertex can only be in one clique
        (it can't be in two cliques), we need to assign each I-vertex to at most
        one group.
        
        Actually, each I-vertex i must appear in exactly |N(i)| cliques if we use
        single-element S, or fewer if we use larger S. The key constraint is that
        each edge (u,i) appears in exactly one clique.
        
        Let me think about this more carefully:
        - Each edge (u,i) where u∈C, i∈I must be in exactly one clique.
        - Each edge (u,v) where u,v∈C must be in exactly one clique.
        
        For I-vertex i with N(i) = {u_1, ..., u_d}:
        We partition N(i) into groups S_1, ..., S_m (each non-empty).
        Each group S_j gives clique {i} ∪ S_j.
        This covers edges (i, s) for s in S_j, AND edges (s1, s2) for s1,s2 in S_j.
        
        The edges within C that are covered by cross-cliques are those where both
        endpoints are in the same S_j for some i.
        
        Remaining C-edges need to be covered by C-only cliques.
        
        To minimize total cliques:
        - We want large S_j (fewer groups per i)
        - But large S_j covers more C-edges, which might reduce C-only cliques needed
        - However, different i's may cover overlapping C-edges, which is fine as long
          as each C-edge is in exactly one clique total.
        
        Wait - each C-edge must be in EXACTLY ONE clique. So if edge (u,v) is in
        clique {i1}∪S_a AND also would be in clique {i2}∪S_b, that's a conflict.
        We need to ensure no C-edge is covered by two different cross-cliques.
        
        This means: for any pair (u,v) in C, at most one I-vertex can have both u
        and v in its assigned groups.
        
        This is getting complex. Let me use a greedy approach:
        
        1. For each i in I (sorted by |N(i)| descending), greedily assign groups.
        2. Track which C-edges have been "claimed" by cross-cliques.
        3. After all cross-cliques, cover remaining C-edges with C-only cliques.
        """
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
        
        # Now cover remaining C-edges with C-only cliques
        remaining_C_edges = C_edges - covered_C_edges
        
        if remaining_C_edges:
            # Greedy: repeatedly find a max clique in the remaining C-edge graph
            # Since C is a clique, any subset of C is a clique.
            # We need to partition remaining_C_edges into cliques.
            # A clique on vertex set S covers all edges within S.
            
            remaining_adj = defaultdict(set)
            for u, v in remaining_C_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)
            
            while remaining_C_edges:
                if time.time() > deadline:
                    break
                
                # Find a good clique: take a vertex with high remaining degree,
                # then find its neighbors in remaining graph, that's a clique
                # (since C is a clique, any subset is a clique)
                
                # Get vertex with max remaining degree
                if not remaining_adj:
                    break
                v = max(remaining_adj.keys(), key=lambda x: len(remaining_adj[x]))
                
                # Clique = {v} ∪ remaining_adj[v]
                S = {v} | remaining_adj[v]
                S_list = sorted(S)
                
                if len(S_list) < 2:
                    # Just take v and one neighbor
                    if remaining_adj[v]:
                        w = next(iter(remaining_adj[v]))
                        S_list = sorted([v, w])
                    else:
                        # Shouldn't happen
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
                # Clean up empty entries
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
