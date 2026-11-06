"""Spectral construction + local swaps for C6-free graphs on 40 vertices."""
import numpy as np
from itertools import combinations
import random

def _has_c6(adj, n, max_iter=100000):
    """Check if the graph with adjacency matrix adj (n x n boolean) contains a C6.
    Uses a systematic check: for each pair of non-adjacent vertices (u,v),
    check if there are two common neighbors a,b that are non-adjacent,
    and then check if there's a vertex c adjacent to both a and b
    (c != u, c != v) completing a C6."""
    # A C6: u-a-b-c-d-v-u where all vertices distinct
    # Equivalent: exist u,v non-adjacent, a,b common neighbors of u,v,
    # a,b non-adjacent, and exist c adjacent to both a and b, c not in {u,v}
    
    # More efficient: for each pair (a,b) of non-adjacent vertices,
    # count common neighbors. If >= 2, check if any two common neighbors
    # are non-adjacent and form a path that creates a C6.
    
    # Actually, the cleanest O(n^4) check:
    # For each 6-tuple of distinct vertices forming a cycle, check adjacency.
    # But that's O(n^6) which is too slow.
    
    # Better: A C6 exists iff there exist vertices u, v, w, x, y, z (all distinct)
    # such that u-v, v-w, w-x, x-y, y-z, z-u are all edges.
    
    # Efficient approach: For each edge (u,v), look for paths of length 5 from u to v
    # that don't use edge (u,v). But this is still expensive.
    
    # Let's use: for each pair of non-adjacent vertices (u,v),
    # let S = N(u) ∩ N(v). For each pair (a,b) in S with a,b non-adjacent,
    # check if N(a) ∩ N(b) contains a vertex c not in {u,v,a,b} that is
    # also adjacent to... wait, this doesn't quite work.
    
    # Correct approach: A C6 through non-edge (u,v) exists iff
    # there exist a ∈ N(u), b ∈ N(v), a ≠ b, a-b not an edge,
    # and c ∈ N(a) ∩ N(b), c ∉ {u,v,a,b}, such that
    # d ∈ N(c) ∩ N(v), d ∉ {u,v,a,b,c}... this is getting complicated.
    
    # Let me use a different characterization:
    # C6 exists iff ∃ distinct u1,...,u6 with edges u1-u2, u2-u3, ..., u6-u1.
    
    # For n=40, let's do a bounded search:
    # For each pair (u,v) non-adjacent, find all paths of length 5 from u to v.
    # If any such path exists with all distinct vertices, we have a C6.
    
    # Path of length 5: u - a - b - c - d - v
    # u adjacent to a, a to b, b to c, c to d, d to v
    # All 6 vertices distinct.
    
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                continue  # Skip adjacent pairs
            # Find all paths u-a-b-c-d-v with distinct vertices
            nu = np.where(adj[u])[0]
            nv = np.where(adj[v])[0]
            for a in nu:
                if a == v:
                    continue
                na = np.where(adj[a])[0]
                for b in na:
                    if b in (u, v, a):
                        continue
                    nb = np.where(adj[b])[0]
                    for c in nb:
                        if c in (u, v, a, b):
                            continue
                        nc = np.where(adj[c])[0]
                        for d in nc:
                            if d in (u, v, a, b, c):
                                continue
                            if adj[d, v]:
                                return True
    return False

def _count_edges(adj, n):
    return int(np.sum(adj)) // 2

def construct():
    n = 40
    
    # Spectral construction: Start with a strongly regular graph-like structure.
    # For n=40, we can use a modified Paley-like or conference graph approach.
    # 
    # Key insight: C6-free graphs have bounded eigenvalues. We want a graph with
    # high edge count but no C6. Start with a sparse base and add edges greedily.
    #
    # Strategy:
    # 1. Start with a 5-regular circulant graph (connect i to i±1, i±2, i±3 mod 40
    #    but only some to get degree 5). Actually, let's use a different approach.
    #
    # 2. Use a greedy construction with C6 checking.
    #
    # 3. Apply local swaps to improve edge count.
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Initial construction: Use a structured base graph.
    # Connect vertex i to i+1, i+2, i+3 (mod 40) for a 3-regular cycle-like graph.
    # Then add more edges greedily.
    
    # Actually, let's start with a sparser base and build up.
    # Use a union of small cliques as base, then add inter-clique edges.
    
    # Better approach: Start empty and add edges in a structured order.
    # Use a greedy algorithm with random restarts.
    
    best_edges = []
    best_count = 0
    
    for seed in [42, 123, 456, 789, 101112]:
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        
        # Generate all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        # Greedily add edges if no C6 is created
        for u, v in all_edges:
            if not adj[u, v]:
                # Check if adding (u,v) creates a C6
                if not _creates_c6(adj, n, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
        
        # Try local swaps: remove an edge, try to add two new edges
        improved = True
        while improved:
            improved = False
            edge_count = _count_edges(adj, n)
            
            # Try removing each edge and adding a different one
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v]:
                        # Temporarily remove
                        adj[u, v] = False
                        adj[v, u] = False
                        
                        # Try to add a new edge
                        for w in range(n):
                            for x in range(w+1, n):
                                if w == u or w == v or x == u or x == v:
                                    continue
                                if not adj[w, x]:
                                    if not _creates_c6(adj, n, w, x):
                                        adj[w, x] = True
                                        adj[x, w] = True
                                        new_count = _count_edges(adj, n)
                                        if new_count > edge_count:
                                            # Keep the swap
                                            improved = True
                                        else:
                                            # Revert
                                            adj[w, x] = False
                                            adj[x, w] = False
                                        break
                            else:
                                continue
                            break
                        
                        # If we didn't improve, put the edge back
                        if not improved:
                            adj[u, v] = True
                            adj[v, u] = True
        
        edge_count = _count_edges(adj, n)
        if edge_count > best_count:
            best_count = edge_count
            best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u, v]]
    
    # Verify the best solution has no C6
    final_adj = np.zeros((n, n), dtype=bool)
    for u, v in best_edges:
        final_adj[u, v] = True
        final_adj[v, u] = True
    
    if _has_c6(final_adj, n):
        # Fallback: use a known valid construction
        # Return a safe graph with fewer edges
        return _safe_construction()
    
    return best_edges

def _creates_c6(adj, n, u, v):
    """Check if adding edge (u,v) to the graph creates a C6.
    A new C6 must use the new edge (u,v). So we look for a path of length 5
    from u to v in the graph without using edge (u,v), with all distinct vertices."""
    # Path: u - a - b - c - d - v, all distinct, none of these edges is (u,v)
    
    # u's neighbors (excluding v since we're checking if adding (u,v) creates C6):
    nu = np.where(adj[u])[0]
    nu = [x for x in nu if x != v]
    
    # v's neighbors (excluding u):
    nv = np.where(adj[v])[0]
    nv = [x for x in nv if x != u]
    
    for a in nu:
        for d in nv:
            if a == d:
                continue
            # Find b adjacent to a (b != u, b != v, b != a, b != d)
            na = np.where(adj[a])[0]
            na = [x for x in na if x not in (u, v, a, d)]
            for b in na:
                # Find c adjacent to b and d (c not in {u, v, a, b, d})
                nb = np.where(adj[b])[0]
                nd = np.where(adj[d])[0]
                common = set(nb) & set(nd)
                common.discard(u)
                common.discard(v)
                common.discard(a)
                common.discard(b)
                common.discard(d)
                for c in common:
                    if len({u, a, b, c, d, v}) == 6:
                        return True
    return False

def _safe_construction():
    """Return a known C6-free graph on 40 vertices."""
    # Use a sparse construction: disjoint union of K5s gives 80 edges, C6-free.
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u = 5*k + a
                v = 5*k + b
                edges.append([u, v])
    return edges
