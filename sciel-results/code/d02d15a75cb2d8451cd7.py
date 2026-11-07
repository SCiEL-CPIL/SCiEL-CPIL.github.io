"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Edges in K
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Identify K and I
        K = list(range(k))
        I = list(range(k, n))
        K_set = set(K)
        I_set = set(I)
        
        # Build adjacency
        # cross_adj[v] = set of K vertices connected to I vertex v
        cross_adj = {v: set() for v in I}
        for v in I:
            for u in K:
                if (u, v) in edges or (v, u) in edges:
                    cross_adj[v].add(u)
        
        # We need to partition all edges into cliques.
        # Strategy: C-C edge-centric approach
        # 1. For each K-K edge (u,w), try to cover it with a triangle {u, v, w} where v is an I-vertex
        #    connected to both u and w. This covers the K-K edge and two cross-edges at once.
        # 2. Remaining cross-edges are covered by 2-cliques {u, v} or by grouping multiple K vertices
        #    with the same I vertex.
        # 3. Remaining K-K edges are covered by pure K cliques.
        
        # Track which edges are covered
        covered_kk_edges = set()  # K-K edges already covered
        covered_cross_edges = set()  # Cross edges already covered
        # For each I vertex, track which K vertices are still "available" (cross-edge not yet covered)
        available_k_for_i = {v: set(cross_adj[v]) for v in I}
        
        # Step 1: Cover K-K edges with triangles (mixed cliques of size 3)
        # For each K-K edge, find an I-vertex connected to both endpoints
        kk_edges = [(u, w) for u in K for w in K if u < w]
        
        # Sort K-K edges by number of common I-neighbors (descending) to prioritize
        # An edge (u,w) can be covered by triangle {u,v,w} iff v is connected to both u and w
        # i.e., v in cross_adj[u] ∩ cross_adj[w]... wait, cross_adj is for I vertices
        # v in I is connected to u iff u in cross_adj[v]
        # So v can form triangle with (u,w) iff u in cross_adj[v] and w in cross_adj[v]
        
        # For each K-K edge, compute how many I vertices can form a triangle
        edge_i_candidates = {}
        for (u, w) in kk_edges:
            candidates = []
            for v in I:
                if u in cross_adj[v] and w in cross_adj[v]:
                    candidates.append(v)
            edge_i_candidates[(u, w)] = candidates
        
        # Process K-K edges: greedily assign I-vertices to cover them
        # We want to minimize the number of cliques, so we want to maximize the use of triangles
        # and then group remaining cross-edges efficiently
        
        # Greedy: for each K-K edge, pick an I-vertex that can cover it
        # But we need to be careful: using an I-vertex for one K-K edge means that
        # cross-edge is "used up" for that I-vertex
        
        # Better approach: 
        # For each I-vertex v, we can form multiple cliques with v.
        # A clique with v and a set S of K vertices covers:
        #   - All K-K edges within S
        #   - All cross-edges (u,v) for u in S
        # We want to partition the K-K edges and cross-edges efficiently.
        
        # Let's think of it as: for each I-vertex v, we choose a collection of subsets
        # of its K-neighbors, and these subsets must be such that no K-K edge is in
        # two different subsets (across all I-vertices and across subsets for the same v).
        
        # Greedy heuristic:
        # 1. For each I-vertex v, start with all its K-neighbors as one set.
        # 2. If this set has K-K edges that conflict with other I-vertices' sets, split.
        
        # Alternative cleaner approach:
        # Phase 1: For each K-K edge, assign it to at most one I-vertex (forming a triangle).
        #          This is a matching-like problem.
        # Phase 2: For each I-vertex, the remaining K-neighbors form cliques with v.
        # Phase 3: Remaining K-K edges form pure K cliques.
        
        # Let's do a simpler greedy:
        # For each I-vertex v (in some order), greedily form the largest possible clique
        # with v such that no K-K edge in that clique is already covered.
        
        # Order I-vertices by degree (descending) - higher degree I-vertices first
        I_sorted = sorted(I, key=lambda v: len(cross_adj[v]), reverse=True)
        
        cliques = []
        covered_kk = set()  # K-K edges covered so far
        
        for v in I_sorted:
            if time.time() > deadline - 5:
                break
            
            # For this I-vertex v, we want to partition N_K(v) into subsets
            # such that no K-K edge appears in two subsets (globally).
            # Greedy: repeatedly pick the largest subset S of remaining N_K(v)
            # such that all K-K edges in S are not yet covered.
            
            remaining = set(cross_adj[v])
            while remaining:
                if time.time() > deadline - 5:
                    break
                
                # Find largest subset S of remaining where all K-K edges are uncovered
                # Greedy: start with one vertex, add others if compatible
                # Compatibility: for all pairs in S, the K-K edge is not in covered_kk
                
                # Pick a starting vertex
                start = next(iter(remaining))
                S = {start}
                remaining.discard(start)
                
                # Try to add other vertices
                for u in list(remaining):
                    can_add = True
                    for w in S:
                        edge = (min(u, w), max(u, w))
                        if edge in covered_kk:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                        remaining.discard(u)
                
                if len(S) >= 1:
                    # Form clique {v} ∪ S
                    clique_verts = list(S) + [v]
                    if len(clique_verts) >= 2:
                        cliques.append(clique_verts)
                        # Mark K-K edges in S as covered
                        for u, w in combinations(S, 2):
                            covered_kk.add((min(u, w), max(u, w)))
        
        # Step 2: Cover remaining K-K edges with pure K cliques
        # Find all K-K edges not yet covered
        remaining_kk_edges = set()
        for u in K:
            for w in K:
                if u < w:
                    if (u, w) not in covered_kk:
                        remaining_kk_edges.add((u, w))
        
        # Greedy clique cover on the graph of remaining K-K edges
        rem_edges = set(remaining_kk_edges)
        while rem_edges:
            if time.time() > deadline - 5:
                break
            
            # Pick an edge and expand to maximal clique
            u, v = next(iter(rem_edges))
            S = {u, v}
            rem_edges.discard((u, v))
            
            changed = True
            while changed:
                changed = False
                for w in K:
                    if w not in S:
                        ok = True
                        for x in S:
                            if (min(w, x), max(w, x)) not in rem_edges:
                                ok = False
                                break
                        if ok:
                            S.add(w)
                            for x in list(S):
                                if x != w:
                                    rem_edges.discard((min(w, x), max(w, x)))
                            changed = True
            
            cliques.append(list(S))
        
        # Verify: every edge must be in exactly one clique
        # For now, trust the logic. In production, add verification.
        
        # Ensure all vertices are ints
        for i, clique in enumerate(cliques):
            cliques[i] = [int(x) for x in clique]
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
