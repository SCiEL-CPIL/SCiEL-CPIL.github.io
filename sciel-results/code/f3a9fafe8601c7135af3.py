import time

def _is_2_colorable(edges, n):
    """
    Check if the hypergraph is 2-colorable.
    For n=9, there are 2^9 = 512 possible 2-colorings.
    We can iterate through all of them.
    """
    # Optimization: If we know the Fano plane is not 2-colorable, we could skip.
    # But for generality and safety, we do a quick check.
    # Actually, for the specific Fano plane, we can just return False.
    # Let's do a proper check to ensure correctness if the edges change.
    
    # Map edges to a set for faster lookup? No, direct check is fine for 7 edges.
    
    for mask in range(1 << n):
        valid = True
        for e in edges:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 10s safety margin
    
    # The Fano plane is the unique 3-uniform hypergraph with 7 edges that is 3-chromatic.
    # Any 3-uniform hypergraph with <= 6 edges is 2-colorable.
    # Thus, 7 is the minimum possible number of edges.
    
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Valid 3-coloring for Fano plane
    # Vertices 7 and 8 are isolated, assign color 0
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify the solution is valid (3-colorable and not 2-colorable)
    # Check 3-coloring validity
    is_3_colorable = True
    for e in fano_edges:
        if fano_colors[e[0]] == fano_colors[e[1]] == fano_colors[e[2]]:
            is_3_colorable = False
            break
            
    if not is_3_colorable:
        # Fallback: This should never happen for the hardcoded Fano plane
        # But if it does, we need a valid output. 
        # We can't easily generate another 7-edge one instantly.
        # We'll just return the Fano plane as is, trusting the hardcoded values.
        pass

    # Check 2-colorability
    # We know Fano is not 2-colorable. 
    # _is_2_colorable takes < 1ms for n=9.
    is_2_colorable = _is_2_colorable(fano_edges, 9)
    
    if is_2_colorable:
        # If the Fano plane were somehow 2-colorable (impossible), 
        # we would need a different hypergraph.
        # Since it's impossible, this block is dead code for the Fano plane.
        pass

    # Ensure types are correct
    edges_list = [list(e) for e in fano_edges]
    colors_list = [int(c) for c in fano_colors]
    
    return {
        'edges': edges_list,
        'colors': colors_list
    }
