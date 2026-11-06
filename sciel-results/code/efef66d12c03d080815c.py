def _try_add_edge_breaking_ind6(adj, n, ind6_set):
    # ind6_set is a list of 6 vertices
    # Try all 15 edges
    for i in range(6):
        for j in range(i + 1, 6):
            u = ind6_set[i]
            v = ind6_set[j]
            # Check if u, v are already connected
            if (adj[u] >> v) & 1:
                continue
            
            # Tentatively add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if this creates a K4
            if not _has_k4(adj, n):
                return True # Edge added successfully
            
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return False
