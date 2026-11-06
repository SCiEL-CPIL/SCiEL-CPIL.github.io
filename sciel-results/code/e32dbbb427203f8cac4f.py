import time
import random

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj_from_colors(colors, n):
    """Build adjacency bitsets from a color list."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_triangles(adj, n):
    """Count total monochromatic triangles."""
    count = 0
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[c][i] >> j) & 1:
                    # Count common neighbors k > j
                    common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                    count += common.bit_count()
    return count

def _find_triangle(adj, n):
    """Find one monochromatic triangle. Returns (i, j, k, c) or None."""
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[c][i] >> j) & 1:
                    common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                    if common:
                        k_bit = common & (-common)
                        k = k_bit.bit_length() - 1
                        return (i, j, k, c)
    return None

def _greedy_extend(base_colors, base_n, target_n):
    """
    Extend a valid coloring of K_base_n to K_target_n.
    Adds vertices base_n to target_n-1 one by one.
    """
    n = target_n
    colors = [0] * (n * (n - 1) // 2)
    
    # Copy base colors
    for i in range(base_n):
        for j in range(i + 1, base_n):
            idx = _compute_index(n, i, j)
            colors[idx] = base_colors[_compute_index(base_n, i, j)]
            
    # Initialize adjacency for the growing graph
    adj = [[0] * n for _ in range(4)]
    for i in range(base_n):
        for j in range(i + 1, base_n):
            c = colors[_compute_index(n, i, j)]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
    # Add new vertices
    for v in range(base_n, n):
        # For each u < v, choose color for (u, v)
        # We want to minimize new triangles involving v.
        # A triangle (u, w, v) is monochromatic if (u,w), (u,v), (w,v) have same color.
        # When we assign color c to (u,v), we create triangles with w < u if (u,w) is c and (w,v) is c.
        # But (w,v) is not assigned yet if w < u < v? No, we assign edges to v one by one.
        # Let's assign edges (u, v) for u = v-1 down to 0? Or 0 to v-1?
        # If we go u = 0 to v-1, then when assigning (u,v), edges (w,v) for w<u are already set.
        # So we can count triangles (u, w, v) where w < u.
        
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                # Count w < u such that (u,w) is c and (w,v) is c
                # adj[c][u] has bits for neighbors of u in color c.
                # We need w < u. So mask with (1 << u) - 1.
                # We also need (w,v) to be c. But (w,v) is not in adj yet for w<u?
                # Wait, if we iterate u from 0 to v-1, then when we are at u,
                # edges (w,v) for w < u have already been assigned and added to adj.
                # So we can check adj[c][w] for bit v? No, adj[c][w] has bit v set if (w,v) is c.
                
                # Let's re-evaluate the order.
                # If we assign (u,v) for u=0..v-1:
                # When assigning (u,v), edges (w,v) for w<u are known.
                # Triangles formed: (u, w, v) with w < u.
                # Condition: color(u,w) == c AND color(w,v) == c.
                
                cost = 0
                for w in range(u):
                    # Check color of (u,w)
                    idx_uw = _compute_index(n, u, w) if u > w else _compute_index(n, w, u)
                    # Actually _compute_index handles i<j.
                    if u > w:
                        c_uw = colors[_compute_index(n, w, u)]
                    else:
                        c_uw = colors[_compute_index(n, u, w)]
                    
                    # Check color of (w,v)
                    if w < v:
                        idx_wv = _compute_index(n, w, v)
                        c_wv = colors[idx_wv]
                    else:
                        c_wv = -1 # Should not happen since w < u < v
                        
                    if c_uw == c and c_wv == c:
                        cost += 1
                        
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
                    
            # Assign best_c to (u, v)
            idx_uv = _compute_index(n, u, v)
            colors[idx_uv] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
            
    return colors

def _local_search_bitset(colors, n, time_budget):
    """
    Local search using bitsets.
    """
    start_time = time.time()
    adj = _build_adj_from_colors(colors, n)
    tri_count = _count_triangles(adj, n)
    
    if tri_count == 0:
        return colors
        
    best_colors = colors[:]
    best_tri = tri_count
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while True:
        if time.time() - start_time > time_budget:
            break
        if T < T_min:
            break
            
        # Find a triangle
        tri = _find_triangle(adj, n)
        if tri is None:
            return colors
            
        i, j, k, c = tri
        
        # Choose an edge to recolor. (i,j), (i,k), or (j,k).
        # Let's pick (i,j) for now, or randomize.
        edges = [(i, j), (i, k), (j, k)]
        u, v = random.choice(edges)
        if u > v:
            u, v = v, u
            
        # Current color of (u,v)
        idx_uv = _compute_index(n, u, v)
        old_c = colors[idx_uv]
        
        # Try a new color
        new_c = random.randint(0, 3)
        while new_c == old_c:
            new_c = random.randint(0, 3)
            
        # Calculate delta in triangle count
        # Removing edge (u,v) from old_c removes triangles where (u,v) is part of a mono-triangle in old_c.
        # Number of such triangles = popcount(adj[old_c][u] & adj[old_c][v])
        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
        
        # Adding edge (u,v) to new_c adds triangles where (u,v) is part of a mono-triangle in new_c.
        # Number of such triangles = popcount(adj[new_c][u] & adj[new_c][v])
        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
        
        delta = added - removed
        
        # Accept if delta < 0 or with probability exp(-delta/T)
        accept = False
        if delta <= 0:
            accept = True
        else:
            if random.random() < pow(2.71828, -delta / T):
                accept = True
                
        if accept:
            # Update bitsets
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            colors[idx_uv] = new_c
            tri_count += delta
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = colors[:]
                if tri_count == 0:
                    return colors
        else:
            # Even if rejected, we might want to try a different move or just cool down
            pass
            
        T *= cooling
        
    return best_colors

def construct():
    try:
        # Deadline guard
        if time.time() - _START_TIME > _DEADLINE - 5:
            return _seed_32()
            
        best_result = _seed_32()
        best_n = 32
        
        # Try to find n=50
        target_n = 50
        time_for_search = _DEADLINE - (time.time() - _START_TIME) - 5
        
        if time_for_search > 10:
            # Greedy extension from 32 to 50
            base = _seed_32()
            colors_50 = _greedy_extend(base['colors'], 32, 50)
            
            # Local search on 50
            improved = _local_search_bitset(colors_50, 50, time_for_search - 2)
            adj_check = _build_adj_from_colors(improved, 50)
            if _count_triangles(adj_check, 50) == 0:
                return {'n': 50, 'colors': improved}
            else:
                # If not zero, check if we can find a valid smaller n?
                # Or just return 32 if 50 is invalid.
                # But maybe we can do a quick search for n=40, 45?
                # For now, fallback to 32 is safer than returning invalid 50.
                pass
        
        # Fallback: try random search for n=40?
        # If time permits, try n=40
        if time.time() - _START_TIME < _DEADLINE - 20:
             target_n_40 = 40
             time_left = _DEADLINE - (time.time() - _START_TIME) - 10
             if time_left > 5:
                 # Greedy extend 32 to 40
                 base = _seed_32()
                 colors_40 = _greedy_extend(base['colors'], 32, 40)
                 improved_40 = _local_search_bitset(colors_40, 40, time_left)
                 adj_check_40 = _build_adj_from_colors(improved_40, 40)
                 if _count_triangles(adj_check_40, 40) == 0:
                     return {'n': 40, 'colors': improved_40}
                     
        return best_result
    except Exception as e:
        return _seed_32()
