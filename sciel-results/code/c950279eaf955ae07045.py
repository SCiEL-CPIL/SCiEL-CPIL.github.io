import random
import time

def construct():
    """
    Greedy construction with forbidden-pair counting for 5-colourings of K_n
    without monochromatic C4. Maximizes n up to 29.
    Uses bitmask adjacency for speed.
    """
    max_n = 29
    num_colors = 5
    
    # Try to build from largest n down
    for n in range(max_n, 10, -1):
        result = _build_coloring(n, num_colors)
        if result is not None:
            return result
    
    # Fallback to known construction for n=11
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

def _build_coloring(n, num_colors, max_attempts=1000):
    """
    Attempt to build a valid 5-coloring of K_n without monochromatic C4.
    Uses forbidden-pair counting to guide color assignment.
    Returns the result dict if successful, else None.
    """
    start_time = time.time()
    timeout = 115  # Leave buffer for 120s limit
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # colors[i][j] = color of edge (i,j), -1 if unassigned
        # We store colors in a flat list or 2D list? 
        # For output, we need flat. For internal, 2D is easier.
        colors = [[-1] * n for _ in range(n)]
        
        # adj[c][v] is a bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(num_colors)]
        
        # common_count[c][a][b] = number of common neighbors of a and b in color c
        # Symmetric, so common_count[c][a][b] == common_count[c][b][a]
        common_count = [[[0]*n for _ in range(n)] for _ in range(num_colors)]
        
        # Precompute edge list
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        rng = random.Random(attempt * 7919 + 104729)
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)
        
        valid = True
        for (u, v) in shuffled_edges:
            if time.time() - start_time > timeout:
                valid = False
                break
            
            # For each color, compute the "cost" of assigning that color:
            # cost = number of new "at-risk" pairs created (pairs that would go from 0 to 1 common neighbor)
            # If any pair would go to 2+, that color is invalid.
            best_color = -1
            best_cost = float('inf')
            
            for c in range(num_colors):
                cost, feasible = _evaluate_color_bitmask(adj, common_count, u, v, c)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_color = c
            
            if best_color == -1:
                valid = False
                break
            
            c = best_color
            # Assign color c to edge (u, v)
            colors[u][v] = c
            colors[v][u] = c
            
            # Update adjacency bitmasks
            # Set bit v in adj[c][u] and bit u in adj[c][v]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            # Update common_count for affected pairs
            # 1. Pair (u, v):
            # common_count[c][u][v] = popcount(adj[c][u] & adj[c][v])
            # Note: adj[c][u] now has v, adj[c][v] now has u.
            # The intersection includes v? No, v is in adj[c][u], is v in adj[c][v]? No (no self loop).
            # The intersection includes u? No, u is in adj[c][v], is u in adj[c][u]? No.
            # So the intersection is exactly the set of other common neighbors.
            common_uv = (adj[c][u] & adj[c][v]).bit_count()
            common_count[c][u][v] = common_uv
            common_count[c][v][u] = common_uv
            
            # 2. Pairs (v, w) for w in adj[c][u] (excluding v)
            # We need to iterate over the set bits of adj[c][u] BEFORE the update?
            # No, we updated adj[c][u] already.
            # The neighbors of u in color c are the bits set in adj[c][u].
            # One of them is v. We want to update common_count for (v, w) for all w != v.
            # For each such w, u is a common neighbor of v and w.
            # So common_count[c][v][w] increases by 1.
            # But wait, is it just +1?
            # common_count[c][v][w] = |N_c(v) ∩ N_c(w)|.
            # Before adding (u,v): u was not in N_c(v).
            # After adding (u,v): u IS in N_c(v).
            # Is u in N_c(w)? Yes, because w was in N_c(u) (symmetry) and edge (u,w) is color c.
            # So u is a NEW common neighbor for pair (v,w).
            # Are there other changes? No, only edge (u,v) was added.
            # So common_count[c][v][w] += 1.
            
            # To be safe and simple, we can recompute:
            # common_count[c][v][w] = (adj[c][v] & adj[c][w]).bit_count()
            
            # Iterate over bits in adj[c][u]
            mask_u = adj[c][u]
            # We want to process w in mask_u, w != v
            # We can clear the bit for v to iterate over others?
            # Or just check w != v inside.
            
            # Optimization: Instead of iterating all bits, we can just iterate.
            # Degree is small.
            w_mask = mask_u
            while w_mask:
                # Get lowest set bit
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                
                if w == v:
                    continue
                
                # Recompute common_count[c][v][w]
                common_vw = (adj[c][v] & adj[c][w]).bit_count()
                common_count[c][v][w] = common_vw
                common_count[c][w][v] = common_vw
                
            # 3. Pairs (u, w) for w in adj[c][v] (excluding u)
            mask_v = adj[c][v]
            w_mask = mask_v
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                
                if w == u:
                    continue
                
                # Recompute common_count[c][u][w]
                common_uw = (adj[c][u] & adj[c][w]).bit_count()
                common_count[c][u][w] = common_uw
                common_count[c][w][u] = common_uw
        
        if valid:
            # Verify no C4 exists (sanity check)
            # The greedy algorithm should prevent this, but let's be sure.
            # If common_count >= 2 anywhere, it's invalid.
            for c in range(num_colors):
                for a in range(n):
                    for b in range(a+1, n):
                        if common_count[c][a][b] >= 2:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                flat_colors = [colors[i][j] for i in range(n) for j in range(i+1, n)]
                return {'n': n, 'colors': flat_colors}
    
    return None

def _evaluate_color_bitmask(adj, common_count, u, v, c):
    """
    Evaluate the cost of assigning color c to edge (u, v).
    Returns (cost, feasible) where:
      - feasible is False if this would create a C4
      - cost is the number of new "at-risk" pairs (pairs going from 0 to 1 common neighbors)
    """
    # 1. Check pair (u, v)
    # Current common neighbors of u and v in color c:
    # This is (adj[c][u] & adj[c][v]).bit_count()
    # Note: adj[c][u] and adj[c][v] do NOT yet contain each other.
    common_uv = (adj[c][u] & adj[c][v]).bit_count()
    
    if common_uv >= 2:
        return (float('inf'), False)
    
    # 2. Check pairs (v, w) for w in adj[c][u] (w != v)
    # If we assign c to (u,v), u becomes a common neighbor of v and w.
    # So common_count[c][v][w] will increase by 1.
    # If common_count[c][v][w] is currently >= 1, it becomes >= 2 -> C4!
    # If common_count[c][v][w] is currently 0, it becomes 1 -> "at-risk" (cost +1).
    
    cost = 0
    mask_u = adj[c][u]
    while mask_u:
        lsb = mask_u & -mask_u
        w = lsb.bit_length() - 1
        mask_u ^= lsb
        
        if w == v:
            continue
        
        # Check current common_count[c][v][w]
        if common_count[c][v][w] >= 1:
            return (float('inf'), False)
        else:
            cost += 1
            
    # 3. Check pairs (u, w) for w in adj[c][v] (w != u)
    # If we assign c to (u,v), v becomes a common neighbor of u and w.
    # So common_count[c][u][w] will increase by 1.
    # If common_count[c][u][w] is currently >= 1, it becomes >= 2 -> C4!
    # If common_count[c][u][w] is currently 0, it becomes 1 -> "at-risk" (cost +1).
    
    mask_v = adj[c][v]
    while mask_v:
        lsb = mask_v & -mask_v
        w = lsb.bit_length() - 1
        mask_v ^= lsb
        
        if w == u:
            continue
        
        # Check current common_count[c][u][w]
        if common_count[c][u][w] >= 1:
            return (float('inf'), False)
        else:
            cost += 1
            
    return (cost, True)
