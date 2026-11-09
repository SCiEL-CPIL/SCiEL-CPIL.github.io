"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

_START = time.time()
_LIMIT = 115.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Initialize partition: each edge is its own clique (2-clique)
    # This is a valid partition. We will try to merge them.
    # To speed up, we can start with a greedy maximal clique extraction 
    # to get a smaller initial set, then refine. 
    # However, the directive asks for a DIFFERENT strategy.
    # Let's try: 
    # 1. Find all maximal cliques (Bron-Kerbosch with pivot).
    # 2. Use a Greedy Set Cover approach to select a subset of these cliques 
    #    that covers all edges, minimizing the number of cliques.
    #    Wait, the cliques in the partition must be disjoint in terms of edges? 
    #    "every edge must lie in exactly one listed clique". 
    #    Yes, edge-disjoint.
    #    So it's a Partition into Cliques (Packing).
    #    
    #    Strategy: 
    #    1. Generate a pool of candidate cliques (maximal cliques).
    #    2. Greedy Selection: Pick the clique that covers the most *uncovered* edges.
    #       Remove those edges. Repeat.
    #    This is the "Top-Down" greedy peeling but with a pre-computed pool of 
    #    maximal cliques, which is often better than expanding on the fly because 
    #    it considers global structure.
    
    # Generate all maximal cliques using Bron-Kerbosch
    def bron_kerbosch(R, P, X):
        if not P and not X:
            yield R
            return
        # Pivot selection: choose vertex in P|X with max degree in P
        u = None
        max_deg = -1
        P_set = set(P)
        X_set = set(X)
        all_nodes = P_set | X_set
        for v in all_nodes:
            deg = len(P_set & set(adj[v].__iter__() and [i for i in range(n) if adj[v][i]])) # This is slow, let's use list
            # Optimization: precompute neighbors
            pass
            
        # Simpler BK with pivot
        if not P:
            if not X:
                yield R
            return
        
        # Pivot u
        P_list = list(P)
        X_list = list(X)
        all_nodes = set(P_list) | set(X_list)
        if not all_nodes:
            if not X:
                yield R
            return
            
        # Pick pivot with max neighbors in P
        best_u = None
        max_common = -1
        # For small n, we can just pick first or max degree
        for u in all_nodes:
            common = 0
            for v in P_list:
                if adj[u][v]:
                    common += 1
            if common > max_common:
                max_common = common
                best_u = u
        
        if best_u is None:
            return
            
        # Recurse on P \ N(best_u)
        # N(best_u) in P
        neighbors_in_P = [v for v in P_list if adj[best_u][v]]
        non_neighbors = [v for v in P_list if not adj[best_u][v]]
        
        for v in non_neighbors:
            new_R = R + [v]
            new_P = [x for x in P_list if adj[v][x]]
            new_X = [x for x in X_list if adj[v][x]]
            yield from bron_kerbosch(new_R, new_P, new_X)
            P_list.remove(v)
            X_list.append(v)
            
    # This recursive BK might be slow if not optimized. 
    # For n=64, it might be okay if we limit depth or use a heuristic.
    # Let's try a different angle: 
    # Greedy Peeling with "Best First" on clique size is what the current code does.
    # Let's try **Iterated Local Search (ILS)** on the partition.
    # 1. Start with a greedy partition (like current).
    # 2. Move: Pick a random clique, try to split it or merge it with neighbors to reduce count?
    #    Merging two cliques C1, C2 into C1 U C2 is only valid if C1 U C2 is a clique.
    #    This reduces count by 1.
    #    So, we want to find pairs of disjoint cliques in the partition whose union is a clique.
    #    This is a "Merge" operation.
    #    Also, we can "Move" a vertex from C1 to C2 if C2 U {v} is a clique and C1 \ {v} is a clique (or empty).
    #    This doesn't change count directly, but might enable further merges.
    
    # Let's implement a Merge-based Local Search.
    # 1. Get an initial partition using Greedy Maximal Clique Extraction (fast).
    # 2. Repeatedly look for mergeable pairs.
    
    def get_initial_partition():
        remaining_edges = set()
        for u, v in edges:
            if u < v:
                remaining_edges.add((u, v))
            else:
                remaining_edges.add((v, u))
        
        partition = []
        while remaining_edges:
            if not _deadline_ok():
                break
            # Find edge with max common neighbors in remaining graph
            best_edge = None
            best_common = -1
            # Sample if too many edges? n=64, E~1100. O(E*n) is fine.
            for e in remaining_edges:
                u, v = e
                common = 0
                for w in range(n):
                    if w == u or w == v: continue
                    if (u, w) in remaining_edges or (w, u) in remaining_edges:
                        if (v, w) in remaining_edges or (w, v) in remaining_edges:
                            common += 1
                if common > best_common:
                    best_common = common
                    best_edge = e
            
            if not best_edge:
                break
                
            u, v = best_edge
            clique = [u, v]
            # Expand
            candidates = []
            for w in range(n):
                if w == u or w == v: continue
                if (u, w) in remaining_edges or (w, u) in remaining_edges:
                    if (v, w) in remaining_edges or (w, v) in remaining_edges:
                        candidates.append(w)
                        
            while candidates:
                best_v = -1
                best_deg = -1
                valid = []
                for w in candidates:
                    ok = True
                    for c in clique:
                        if not (adj[c][w]):
                            ok = False
                            break
                    if ok:
                        valid.append(w)
                if not valid:
                    break
                for w in valid:
                    deg = 0
                    for x in valid:
                        if x != w and adj[w][x]:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = w
                if best_v == -1: break
                clique.append(best_v)
                candidates = [w for w in candidates if adj[w][best_v]]
                
            partition.append(clique)
            # Remove edges
            for a in clique:
                for b in clique:
                    if a < b:
                        remaining_edges.discard((a, b))
                    else:
                        remaining_edges.discard((b, a))
                        
        # Add remaining edges as 2-cliques if deadline hit or stuck
        if remaining_edges:
            for e in remaining_edges:
                partition.append(list(e))
                
        return partition

    partition = get_initial_partition()
    
    # Local Search: Merge
    # A merge is valid if C1 U C2 is a clique.
    # We want to find such pairs.
    
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if not adj[vertices[i]][vertices[j]]:
                    return False
        return True

    improved = True
    while improved and _deadline_ok():
        improved = False
        # Sort cliques by size descending to try merging larger ones first?
        # Or just iterate.
        n_cliques = len(partition)
        for i in range(n_cliques):
            if not _deadline_ok(): break
            for j in range(i + 1, n_cliques):
                c1 = partition[i]
                c2 = partition[j]
                # Check if c1 U c2 is a clique
                # Since c1 and c2 are disjoint sets of vertices (partition of edges implies vertex sets can overlap? 
                # Wait. "Clique partition" usually means edge partition. Vertices can be in multiple cliques.
                # Example: Triangle (0,1,2). Edges (0,1), (1,2), (0,2).
                # If we have clique [0,1] and [1,2], they share vertex 1.
                # Union is {0,1,2}. Is it a clique? Yes.
                # So we can merge [0,1] and [1,2] into [0,1,2].
                # This reduces count from 2 to 1.
                
                union = set(c1) | set(c2)
                if is_clique(list(union)):
                    # Merge
                    partition[i] = list(union)
                    partition.pop(j)
                    improved = True
                    # Restart inner loop or break to re-index
                    break
            if improved:
                break
                
    return partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
