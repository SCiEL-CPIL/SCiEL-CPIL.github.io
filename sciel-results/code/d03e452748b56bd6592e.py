"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 20
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Precompute all potential edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    import random
    random.seed(42)
    random.shuffle(all_edges)
    
    # For each potential edge (u, v), we need to check if adding it would violate K2,3.
    # Adding edge (u, v) means:
    # - For every w != u, v: new common neighbors of (u, w) = old_common(u,w) + (1 if adj[w,v] else 0)
    # - For every w != u, v: new common neighbors of (v, w) = old_common(v,w) + (1 if adj[w,u] else 0)
    # We need both to be <= 2.
    
    # Vectorized approach: maintain common neighbor counts matrix
    # common[u, w] = number of common neighbors of u and w
    common = np.zeros((n, n), dtype=np.int8)
    
    for (u, v) in all_edges:
        # Check if adding (u, v) is valid
        # For all w != u, v:
        #   new_common_uw = common[u, w] + (1 if adj[w, v] else 0)
        #   new_common_vw = common[v, w] + (1 if adj[w, u] else 0)
        # Both must be <= 2
        
        # Create mask for w != u, v
        mask = np.ones(n, dtype=bool)
        mask[u] = False
        mask[v] = False
        
        # Compute new common neighbors for pairs (u, w) for all w in mask
        # new_common_uw = common[u, mask] + adj[mask, v]
        new_common_uw = common[u, mask] + adj[mask, v]
        
        # Compute new common neighbors for pairs (v, w) for all w in mask
        # new_common_vw = common[v, mask] + adj[mask, u]
        new_common_vw = common[v, mask] + adj[mask, u]
        
        # Check if any exceed 2
        if np.any(new_common_uw > 2) or np.any(new_common_vw > 2):
            continue
        
        # Valid - add the edge
        adj[u, v] = 1
        adj[v, u] = 1
        
        # Update common neighbor counts
        # Adding edge (u, v) increases common neighbors for:
        # - Pairs (u, w) where w is a neighbor of v (w != u): common[u, w] += 1
        # - Pairs (v, w) where w is a neighbor of u (w != v): common[v, w] += 1
        # - Pair (u, v): common[u, v] += 1 (they now share... wait, no)
        # Actually, common[u, w] counts neighbors of u that are also neighbors of w.
        # When we add edge (u, v), v becomes a neighbor of u.
        # So for any w that is a neighbor of v (and w != u), v is now a common neighbor of u and w.
        # Thus common[u, w] += 1 for all w in N(v) \ {u}
        # Similarly, u becomes a neighbor of v, so common[v, w] += 1 for all w in N(u) \ {v}
        
        # Get neighbors of v (excluding u)
        neighbors_of_v = np.where(adj[v, :] == 1)[0]
        neighbors_of_v = neighbors_of_v[neighbors_of_v != u]
        
        # Get neighbors of u (excluding v)
        neighbors_of_u = np.where(adj[u, :] == 1)[0]
        neighbors_of_u = neighbors_of_u[neighbors_of_u != v]
        
        # Update common counts
        if len(neighbors_of_v) > 0:
            common[u, neighbors_of_v] += 1
            common[neighbors_of_v, u] += 1
        
        if len(neighbors_of_u) > 0:
            common[v, neighbors_of_u] += 1
            common[neighbors_of_u, v] += 1
    
    # Convert adjacency matrix to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
