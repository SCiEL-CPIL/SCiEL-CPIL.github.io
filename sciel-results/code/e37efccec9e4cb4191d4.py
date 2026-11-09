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
    # We allocate more time to larger n initially, but if they fail quickly, we move on.
    # 26 is the known max. 27/28/29 are unlikely but we must try.
    for n in range(max_n, 10, -1):
        # Give larger n slightly more attempts if we have time, but keep it bounded.
        # Actually, just use a fixed timeout per n.
        result = _build_coloring(n, num_colors, timeout=15.0)
        if result is not None:
            return result
    
    # Fallback to known construction for n=11 (should not happen if 26 works)
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

def _build_coloring(n, num_colors, timeout=15.0):
    """
    Attempt to build a valid 5-coloring of K_n without monochromatic C4.
    Uses forbidden-pair counting to guide color assignment.
    Returns the result dict if successful, else None.
    """
    start_time = time.time()
    
    # Precompute edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # We will try multiple random orderings
    # Number of attempts depends on time available
    max_attempts = 5000 # High number, but we break on timeout
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        rng = random.Random(attempt * 7919 + 104729 + n)
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)
        
        # adj[c][v] is a bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(num_colors)]
        # deg[c][v] is the degree of v in color c
        deg = [[0] * n for _ in range(num_colors)]
        
        # colors[i][j] = color of edge (i,j)
        # We only need to store the final result, so we can just build the list at the end
        # But we need to know if we succeeded.
        
        valid = True
        assigned_colors = [] # To build the final list, we need to map back to lexicographic order
        # Actually, it's easier to keep a 2D array or a dict, then flatten at the end.
        # Or, since we iterate in a specific order, we can just store (u,v,c) and sort later?
        # No, we must output in lexicographic order (i<j).
        # Let's keep a 2D list for simplicity, it's small (29x29).
        colors = [[-1] * n for _ in range(n)]
        
        for (u, v) in shuffled_edges:
            # Check timeout
            if time.time() - start_time > timeout:
                valid = False
                break
            
            best_color = -1
            best_cost = float('inf')
            
            for c in range(num_colors):
                cost, feasible = _evaluate_color(adj, deg, u, v, c)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_color = c
            
            if best_color == -1:
                valid = False
                break
            
            c = best_color
            colors[u][v] = c
            colors[v][u] = c
            
            # Update state
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            deg[c][u] += 1
            deg[c][v] += 1
        
        if valid:
            # Verify
            if _verify(adj, n, num_colors):
                flat_colors = [colors[i][j] for i in range(n) for j in range(i + 1, n)]
                return {'n': n, 'colors': flat_colors}
    
    return None

def _evaluate_color(adj, deg, u, v, c):
    """
    Evaluate cost of assigning color c to edge (u, v).
    Returns (cost, feasible).
    """
    # Check if (u, v) already has 2 common neighbors in color c
    common_uv = (adj[c][u] & adj[c][v]).bit_count()
    if common_uv >= 2:
        return (float('inf'), False)
    
    cost = 0
    
    # Check neighbors of u in color c
    # For each w in N_c(u), w != v:
    # u becomes a common neighbor of v and w.
    # If v and w already have a common neighbor in c, adding u makes it 2 -> C4.
    # If they have 0, it becomes 1 -> Risk.
    
    mask_u = adj[c][u]
    # We iterate over bits in mask_u
    m = mask_u
    while m:
        lsb = m & -m
        w = lsb.bit_length() - 1
        m ^= lsb
        
        if w == v:
            continue
            
        # Check common neighbors of v and w in color c
        # Note: adj[c][v] and adj[c][w] do NOT include u yet.
        # So the current count is (adj[c][v] & adj[c][w]).bit_count()
        common_vw = (adj[c][v] & adj[c][w]).bit_count()
        
        if common_vw >= 1:
            return (float('inf'), False)
        else:
            cost += 1

    # Check neighbors of v in color c
    # For each w in N_c(v), w != u:
    # v becomes a common neighbor of u and w.
    
    mask_v = adj[c][v]
    m = mask_v
    while m:
        lsb = m & -m
        w = lsb.bit_length() - 1
        m ^= lsb
        
        if w == u:
            continue
            
        # Check common neighbors of u and w in color c
        common_uw = (adj[c][u] & adj[c][w]).bit_count()
        
        if common_uw >= 1:
            return (float('inf'), False)
        else:
            cost += 1
    
    # Add a small penalty for high degree to balance colors
    # This helps avoid bottlenecks where one color gets too dense
    cost += 0.01 * (deg[c][u] + deg[c][v])
    
    return (cost, True)

def _verify(adj, n, num_colors):
    """
    Verify no C4 in any color.
    """
    for c in range(num_colors):
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[c][i] & adj[c][j]).bit_count() >= 2:
                    return False
    return True
