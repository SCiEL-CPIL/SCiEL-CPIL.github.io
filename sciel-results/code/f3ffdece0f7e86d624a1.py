import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    # Precomputation
    # 1. All 8-subsets of 0..15
    all_8_subsets = list(combinations(range(n), 8))
    
    # 2. Q3 masks in K8
    # Q3 vertices: 0..7
    # Edges: (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
    # This is one specific labeling.
    # We need all 840 isomorphic copies.
    # Generate all permutations of 0..7 and check if the induced subgraph is Q3?
    # No, generate all subsets of edges of size 12 that form a Q3.
    
    # Simpler: Generate all 840 masks by iterating over all 8! permutations of 0..7
    # and checking if the permutation maps the standard Q3 to a valid subgraph?
    # Actually, the set of edges is invariant under automorphism.
    # The 840 masks are the distinct sets of 12 edges in K8 that form a Q3.
    
    standard_q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    def edge_to_idx(u, v):
        if u > v: u, v = v, u
        # Map (0,1)..(6,7) to 0..27
        # 0: (0,1)
        # 1: (0,2)
        # ...
        # Formula: idx = u*7 - u*(u+1)//2 + (v - u - 1)
        return u * 7 - u * (u + 1) // 2 + (v - u - 1)
        
    standard_mask = 0
    for u, v in standard_q3_edges:
        standard_mask |= 1 << edge_to_idx(u, v)
        
    q3_masks = set()
    # Generate all permutations of 0..7
    # This is 40320 iterations.
    # For each perm, map the standard edges to new edges.
    # If the set of edges is the same as standard, it's an automorphism.
    # We want the distinct masks.
    
    import itertools
    for perm in itertools.permutations(range(8)):
        mask = 0
        valid = True
        for u, v in standard_q3_edges:
            nu, nv = perm[u], perm[v]
            if nu > nv: nu, nv = nv, nu
            idx = edge_to_idx(nu, nv)
            mask |= 1 << idx
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    # Should be 840
    
    # Precompute masks by edge in K8
    # For each edge (i,j) in K8, list of masks that contain it.
    masks_by_edge = { (i,j): [] for i in range(8) for j in range(i+1, 8) }
    for mask in q3_masks_list:
        # Iterate through all 28 edges in K8
        for i in range(8):
            for j in range(i+1, 8):
                idx = edge_to_idx(i, j)
                if (mask >> idx) & 1:
                    masks_by_edge[(i,j)].append(mask)
                    
    # Precompute 6-subsets for each pair (u,v)
    # pair_data[u][v] = list of (T_tuple, ku, kv)
    # T_tuple is the sorted 6-set from V \ {u,v}
    # ku, kv are indices of u, v in sorted(T + (u,v))
    
    pair_data = {}
    for u in range(n):
        for v in range(u+1, n):
            others = [x for x in range(n) if x != u and x != v]
            data = []
            for T in combinations(others, 6):
                S = tuple(sorted(T + (u, v)))
                ku = S.index(u)
                kv = S.index(v)
                data.append((T, ku, kv, S))
            pair_data[(u,v)] = data
            pair_data[(v,u)] = data
            
    # Initialize graph
    # Start with K3,13
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= 1 << b
            adj[b] |= 1 << a
            
    best_edges = 39
    best_adj = adj[:]
    
    def count_edges(adj):
        return sum(bin(x).count("1") for x in adj) // 2
        
    def is_q3_free_after_add(u, v, adj):
        # Check if adding (u,v) creates a Q3
        # Only need to check Q3s containing (u,v)
        data = pair_data[(u,v)]
        for T, ku, kv, S in data:
            # Build local mask for G[S]
            local_mask = 0
            for i in range(8):
                for j in range(i+1, 8):
                    su = S[i]
                    sv = S[j]
                    # Check if edge (su, sv) is in adj
                    if (adj[su] >> sv) & 1:
                        idx = edge_to_idx(i, j)
                        local_mask |= 1 << idx
            
            if bin(local_mask).count("1") < 12:
                continue
                
            # Check if any Q3 mask containing edge (ku, kv) is a subset
            # Note: The edge (ku, kv) in local coords corresponds to (u,v) in global
            # But we haven't added (u,v) to adj yet.
            # So local_mask does NOT contain (ku, kv).
            # We must check if (local_mask | (1 << edge_to_idx(ku, kv))) contains a Q3 mask.
            
            check_mask = local_mask | (1 << edge_to_idx(ku, kv))
            
            for m in masks_by_edge[(ku, kv)]:
                if (check_mask & m) == m:
                    return False
        return True
        
    # Main loop
    while time.time() < deadline:
        # Try to add edges
        adj_curr = best_adj[:]
        
        # Get missing edges
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj_curr[u] >> v) & 1):
                    missing.append((u, v))
                    
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if is_q3_free_after_add(u, v, adj_curr):
                adj_curr[u] |= 1 << v
                adj_curr[v] |= 1 << u
                
        curr_edges = count_edges(adj_curr)
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = adj_curr[:]
            
    # Return edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
