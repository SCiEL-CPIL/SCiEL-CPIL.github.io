<<<<<<< SEARCH
def _construct_known_17():
    """
    Find a valid 4-coloring of K_17 using local search.
    Falls back to smaller n if not found.
    """
    import random
    random.seed(42)
    
    # Try n=17 first with local search
    result = _find_valid_coloring(17, time_limit=15.0)
    if result is not None:
        return {'n': 17, 'colors': result}
    
    # Try n=16
    result = _find_valid_coloring(16, time_limit=10.0)
    if result is not None:
        return {'n': 16, 'colors': result}
    
    # Try n=15
    result = _find_valid_coloring(15, time_limit=10.0)
    if result is not None:
        return {'n': 15, 'colors': result}
    
    # Try n=14
    result = _find_valid_coloring(14, time_limit=10.0)
    if result is not None:
        return {'n': 14, 'colors': result}
    
    # Try n=13
    result = _find_valid_coloring(13, time_limit=10.0)
    if result is not None:
        return {'n': 13, 'colors': result}
    
    # Try n=12
    result = _find_valid_coloring(12, time_limit=10.0)
    if result is not None:
        return {'n': 12, 'colors': result}
    
    # Try n=11
    result = _find_valid_coloring(11, time_limit=10.0)
    if result is not None:
        return {'n': 11, 'colors': result}
    
    # Try n=10
    result = _find_valid_coloring(10, time_limit=10.0)
    if result is not None:
        return {'n': 10, 'colors': result}
    
    # Try n=9
    result = _find_valid_coloring(9, time_limit=5.0)
    if result is not None:
        return {'n': 9, 'colors': result}
    
    # Try n=8
    result = _find_valid_coloring(8, time_limit=5.0)
    if result is not None:
        return {'n': 8, 'colors': result}
    
    # Try n=7
    result = _find_valid_coloring(7, time_limit=5.0)
    if result is not None:
        return {'n': 7, 'colors': result}
    
    # Try n=6
    result = _find_valid_coloring(6, time_limit=5.0)
    if result is not None:
        return {'n': 6, 'colors': result}
    
    # Try n=5
    result = _find_valid_coloring(5, time_limit=2.0)
    if result is not None:
        return {'n': 5, 'colors': result}
    
    # n=4 always works
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}
=======
def _find_valid_coloring(n, time_limit=10.0):
    """
    Find a valid 4-coloring of K_n using targeted local search with C6-counting heuristic.
    Prefers flipping edges that are in C6s.
    """
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    # Build edge index lookup
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        # Count edges before vertex i
        # Vertex k has edges to k+1,...,n-1, so n-1-k edges
        # Edges before vertex i: sum_{k=0}^{i-1} (n-1-k)
        before = 0
        for k in range(i):
            before += (n - 1 - k)
        # Within vertex i's edges, (i,j) is at position j-1-i
        return before + (j - 1 - i)
    
    def count_c6s_and_edges(adj):
        """Count C6s and return (count, set of edges in C6s)."""
        total = 0
        c6_edges = set()
        for c in range(4):
            for v0 in range(n):
                mask_v1 = adj[c][v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    
                    mask_v2 = adj[c][v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        
                        mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            
                            mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                
                                mask_v5 = adj[c][v4] & adj[c][v0]
                                used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used_mask
                                if mask_v5:
                                    total += 1
                                    # Mark edges
                                    edges = [(v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)]
                                    for a, b in edges:
                                        if a > b:
                                            a, b = b, a
                                        c6_edges.add((a, b))
        return total, c6_edges
    
    # Start with random coloring
    random.seed(42)
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    adj = _build_adj(colors, n)
    current_c6s, c6_edges = count_c6s_and_edges(adj)
    
    if current_c6s == 0:
        return colors
    
    best_colors = colors[:]
    best_c6s = current_c6s
    best_c6_edges = c6_edges
    
    # Local search with targeted edge selection
    for iteration in range(200000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors
            
        # Preferentially pick edges in C6s (80% chance if available)
        if c6_edges and random.random() < 0.8:
            # Pick a random edge from c6_edges
            edge_tuple = random.choice(list(c6_edges))
            i, j = edge_tuple
            if i > j:
                i, j = j, i
            edge_idx = get_edge_idx(i, j)
        else:
            edge_idx = random.randint(0, num_edges - 1)
        
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        colors[edge_idx] = new_color
        adj = _build_adj(colors, n)
        new_c6s, new_c6_edges = count_c6s_and_edges(adj)
        
        if new_c6s <= best_c6s:
            best_c6s = new_c6s
            best_colors = colors[:]
            c6_edges = new_c6_edges
        else:
            colors[edge_idx] = old_color
            # Restore adj
            adj = _build_adj(colors, n)
    
    if best_c6s == 0:
        return best_colors
    return None
>>>>>>> REPLACE

