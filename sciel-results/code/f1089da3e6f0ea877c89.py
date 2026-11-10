import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 18
    # Precompute all subsets of size 12
    # Represent subsets as bitmasks for faster intersection
    subsets_12 = []
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_12.append(mask)
        
    # Also need list of vertices in mask for iteration
    # Or just iterate 0..17 and check bit
    
    def is_triangle_free(mask, adj):
        """
        Check if the subgraph induced by 'mask' is triangle-free.
        adj is a list of 18 integers (bitmasks).
        """
        # Iterate over all pairs (u, v) in mask such that u < v
        # If (u, v) is an edge, check if they have a common neighbor in mask
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            # Neighbors of u in mask
            nu = adj[u] & mask
            # Remove u itself if present (it shouldn't be, as no self-loops)
            # Iterate over neighbors v of u in mask
            v_mask = nu
            while v_mask:
                # Extract lowest set bit
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                
                # Check if u and v have a common neighbor in mask
                # Common neighbors of u and v are in adj[u] & adj[v]
                # We need to check if any of these are in mask
                # Actually, we just need to check if (adj[u] & adj[v] & mask) != 0
                # But we must exclude u and v themselves from common neighbors?
                # adj[u] doesn't contain u. adj[v] doesn't contain v.
                # So adj[u] & adj[v] doesn't contain u or v.
                # So we just check if (adj[u] & adj[v] & mask) != 0
                
                if (adj[u] & adj[v] & mask) != 0:
                    return False
        return True

    def has_triangle_free_12(adj):
        for mask in subsets_12:
            if is_triangle_free(mask, adj):
                return True
        return False

    def generate_random_max_k4_free():
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            # Check if adding (u, v) creates a K4
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            
            # Check if there is an edge within 'common'
            # If yes, adding (u,v) creates a K4 with that edge's endpoints
            has_edge_in_common = False
            c_mask = common
            while c_mask:
                lsb = c_mask & -c_mask
                w = lsb.bit_length() - 1
                c_mask ^= lsb
                # Check if w is connected to any other node in common
                # adj[w] & common will contain w's neighbors in common
                # We need to check if adj[w] & common != 0 (excluding w itself? no, adj[w] doesn't have w)
                # But we need to check if there is ANOTHER node x in common such that (w,x) is edge.
                # So adj[w] & common should have at least one bit set.
                # Wait, if adj[w] & common != 0, it means w is connected to some x in common.
                # Since x != w (no self loops), this forms a triangle w-x-...?
                # If (w,x) is an edge, and w,x are common neighbors of u,v,
                # then u-v-w-x-u is a K4?
                # u-v is the new edge.
                # u-w, v-w are edges (common).
                # u-x, v-x are edges (common).
                # w-x is an edge.
                # Yes, this is a K4.
                
                if (adj[w] & common) != 0:
                    has_edge_in_common = True
                    break
            
            if not has_edge_in_common:
                # Safe to add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    best_graph = None
    best_score = 12 # Seed score
    
    # Initial seed graph: K6,6,6
    # Parts: 0-5, 6-11, 12-17
    seed_adj = [0] * n
    for i in range(6):
        for j in range(6, 12):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
        for j in range(12, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
            
    # Verify seed score is 12 (it is, two parts form independent set of size 12)
    # Actually, we just return edges from adj
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    current_best_edges = adj_to_edges(seed_adj)
    
    while time.time() < deadline:
        adj = generate_random_max_k4_free()
        
        # Check if largest triangle-free induced subgraph is < 12
        # i.e., no triangle-free induced subgraph of size 12
        if not has_triangle_free_12(adj):
            # Score is <= 11
            # We can return this immediately as it's better than 12
            return adj_to_edges(adj)
            
        # Optional: try to improve?
        # For now, just keep searching random graphs.
        
    return current_best_edges

# EVOLVE-BLOCK-END
