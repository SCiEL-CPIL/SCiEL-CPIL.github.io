def _optimize_edges(adj, n, deadline):
    """
    Local search to reduce alpha while maintaining triangle-free property.
    Returns (adj, alpha).
    """
    adj = list(adj)
    best_alpha = _alpha_exact_bnb(adj, n, limit=10)
    
    if best_alpha < 10:
        return adj, best_alpha

    # Pre-compute valid non-edges? 
    # It's better to generate them on the fly or in a list.
    
    while time.time() < deadline:
        improved = False
        
        # Generate list of candidate edges that can be added without creating a triangle
        candidates = []
        for i in range(n):
            # Non-neighbors of i
            non_neighbors = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
            # Iterate through non-neighbors j > i to avoid duplicates
            mask = non_neighbors
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if j > i:
                    # Check triangle: adj[i] & adj[j] must be 0
                    if (adj[i] & adj[j]) == 0:
                        candidates.append((i, j))
                mask -= lsb
        
        if not candidates:
            break
            
        random.shuffle(candidates)
        
        # Try to add one edge that improves alpha
        for i, j in candidates:
            if time.time() > deadline:
                return adj, best_alpha
            
            # Add edge
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            new_alpha = _alpha_exact_bnb(adj, n, limit=10)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                improved = True
                if best_alpha < 10:
                    return adj, best_alpha
                # If we found an improvement, we break and restart the candidate generation
                # because the graph has changed
                break
            else:
                # Revert
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                
        if not improved:
            # No single edge addition helps. 
            # We could try edge flipping (remove one, add another) but that's more complex.
            # For now, break to avoid infinite loop on this specific graph instance.
            break
            
    return adj, best_alpha
