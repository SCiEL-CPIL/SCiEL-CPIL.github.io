"""Rewritten program using a Tomita branch-and-bound algorithm for Maximum Independent Set."""
import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate candidate vectors: weight 1, 2, 3
    candidates = []
    indices = range(D)
    
    # Weight 1
    for i in indices:
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(indices, 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i], v[j] = s1, s2
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(indices, 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = s1, s2, s3
                    candidates.append(v)
                    
    n = len(candidates)
    
    # Precompute norms squared
    norms2 = [sum(x*x for x in v) for v in candidates]
    
    # Build conflict graph
    # u and v conflict if ||u-v||^2 < max(||u||^2, ||v||^2)
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # Conflict if ||u||^2 + ||v||^2 - 2 u.v < max(||u||^2, ||v||^2)
    # Let M = max(||u||^2, ||v||^2).
    # If ||u||^2 >= ||v||^2, M = ||u||^2.
    # ||u||^2 + ||v||^2 - 2 u.v < ||u||^2 => ||v||^2 < 2 u.v => u.v > ||v||^2 / 2
    # If ||v||^2 > ||u||^2, M = ||v||^2.
    # ||u||^2 + ||v||^2 - 2 u.v < ||v||^2 => ||u||^2 < 2 u.v => u.v > ||u||^2 / 2
    
    # General condition: u.v > min(||u||^2, ||v||^2) / 2
    
    adj = [set() for _ in range(n)]
    
    # Optimization: Group by norm squared?
    # Or just O(n^2) check? n = 22 + 220 + 1320 = 1562.
    # n^2 ~ 2.4M. Dot products are cheap. This is fine.
    
    for i in range(n):
        ui = candidates[i]
        ni = norms2[i]
        for j in range(i+1, n):
            uj = candidates[j]
            nj = norms2[j]
            dot = 0
            # Compute dot product
            for k in range(D):
                if ui[k] and uj[k]:
                    dot += ui[k] * uj[k]
            
            # Check conflict
            min_norm2 = min(ni, nj)
            threshold = min_norm2 / 2.0
            if dot > threshold:
                adj[i].add(j)
                adj[j].add(i)
                
    # Tomita's Branch and Bound for Maximum Independent Set
    # We want to find the maximum independent set.
    # This is equivalent to finding the maximum clique in the complement graph.
    # However, Tomita's algorithm is usually described for Max Clique.
    # We can run Max Clique on the complement graph.
    # Complement graph edge (i,j) exists if i and j are NOT in conflict.
    
    # Let's implement Max Clique on the complement graph.
    # Nodes: 0..n-1
    # Adjacency list for complement graph:
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
                
    # Tomita's BB-MCS (Bron-Kerbosch with coloring)
    best_clique = []
    
    def expand(P, X, clique):
        nonlocal best_clique
        if not P and not X:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return
        
        # Order P by degree in P (or just use a static order for simplicity, 
        # but coloring bound requires ordering)
        # For Tomita, we color P.
        # We need to find a coloring of P such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the clique size in P.
        
        # Simple coloring heuristic:
        # Greedy coloring of P
        colors = {}
        color_order = []
        remaining = list(P)
        # Sort by degree in P descending to improve coloring?
        # Let's just use a simple greedy coloring on the induced subgraph of P.
        
        # To make it fast, we can use a standard greedy coloring.
        # But we need to integrate it with the recursion.
        # Standard Tomita:
        # 1. Color P.
        # 2. Iterate through vertices in reverse color order.
        
        # Let's implement a helper to color a set of vertices.
        def color_vertices(vertices):
            # vertices is a list of node ids
            # Returns list of (vertex, color) sorted by color descending
            uncolored = list(vertices)
            colors_map = {}
            color_id = 0
            # Simple greedy:
            # We can use a more effective coloring like DSatur, but greedy is okay for start.
            # Actually, Tomita uses a specific ordering.
            # Let's use a simple greedy coloring where we assign the smallest available color.
            # But we want the number of colors to be small (tight bound).
            # So we should assign colors carefully.
            
            # A common approach:
            # Order vertices by degree in the induced subgraph descending.
            # Then greedy color.
            
            # For now, let's just use a simple sequential coloring which might be loose.
            # To be safe, let's implement a basic greedy coloring.
            
            # Actually, the bound is the number of colors.
            # If we use a loose coloring, the bound is loose, and we prune less.
            # Let's try to implement a decent coloring.
            
            # Greedy coloring algorithm:
            # 1. Order vertices.
            # 2. Assign colors.
            
            # Let's just use the node index order for simplicity first, 
            # but that might be bad.
            # Let's use degree in P.
            
            degs = {v: len(comp_adj[v] & set(vertices)) for v in vertices}
            ordered = sorted(vertices, key=lambda v: degs[v], reverse=True)
            
            assigned = {}
            max_color = 0
            for v in ordered:
                # Find smallest color not used by neighbors in `ordered` that are already assigned
                used_colors = set()
                for u in comp_adj[v]:
                    if u in assigned:
                        used_colors.add(assigned[u])
                        
                c = 0
                while c in used_colors:
                    c += 1
                assigned[v] = c
                max_color = max(max_color, c)
                
            # Create list of (v, color)
            result = [(v, c) for v, c in assigned.items()]
            # Sort by color descending
            result.sort(key=lambda x: x[1], reverse=True)
            return result

        colored_P = color_vertices(P)
        
        for v, c in colored_P:
            # If current clique size + remaining colors < best size, prune
            if len(clique) + c < len(best_clique):
                break
                
            # Recurse
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            expand(new_P, new_X, clique + [v])
            
            P.remove(v)
            X.add(v)
            
    # Initial call
    # P = all nodes, X = empty
    all_nodes = set(range(n))
    expand(all_nodes, set(), [])
    
    # best_clique contains node indices of the max independent set in original graph
    # (which is a max clique in complement graph)
    
    # Construct the result array
    result_vectors = [candidates[i] for i in best_clique]
    
    return np.array(result_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
