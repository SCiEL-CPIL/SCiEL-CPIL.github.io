import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 145.0  # Leave 5s safety margin for the 150s limit

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_index(n, i, j):
    """Compute the flat index for edge (i, j) in lexicographic order."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for each color and vertex."""
    # adj[c][v] is a bitmask of vertices u such that edge (v, u) has color c
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    """Count total monochromatic triangles."""
    count = 0
    for c in range(4):
        for i in range(n):
            # Only consider neighbors j > i to avoid double counting and self-loops
            # We iterate j > i, and check common neighbors k > j
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                # Common neighbors of i and j in color c, strictly greater than j
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _greedy_init(n):
    """
    Initialize coloring by adding vertices one by one.
    For each new vertex v, choose the color for edge (u, v) (u < v)
    to minimize the number of new monochromatic triangles formed.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            # We need to estimate the cost of assigning color c to (u, v)
            # Cost = number of w < u such that c(u,w)=c and c(v,w)=c? 
            # No, v is new, so no edges from v to w exist yet in the graph.
            # However, we are building the graph incrementally.
            # If we assign color c to (u, v), we form triangles (u, v, w) where w < u.
            # The edges (u, w) and (v, w) must both be color c.
            # But (v, w) is not assigned yet! 
            # This simple greedy doesn't work directly because (v, w) is unassigned.
            
            # Alternative Greedy:
            # When adding v, we assign colors to (0,v), (1,v), ..., (v-1, v).
            # We can do this sequentially. For each u, we choose c to minimize
            # the number of triangles (u, v, w) where w < u.
            # For a fixed u and proposed color c for (u, v):
            # A triangle (u, v, w) is monochromatic color c if:
            # 1. c(u, v) == c
            # 2. c(u, w) == c
            # 3. c(v, w) == c
            # Since we are assigning (u, v) now, and (v, w) for w < u were assigned in previous steps (when v was the "new" vertex relative to w? No, w < u < v).
            # Wait, when we added u, we assigned (w, u) for w < u.
            # When we add v, we assign (w, v) for w < v.
            # So when deciding (u, v), the edges (w, v) for w < u have ALREADY been assigned?
            # Yes, if we iterate u from 0 to v-1.
            # So for a candidate color c for (u, v):
            # Triangles formed are (u, v, w) for w < u.
            # These are monochromatic c if c(u, w) == c AND c(v, w) == c.
            # c(u, w) is known. c(v, w) is known (assigned in previous steps for this v).
            
            for c in range(4):
                cost = 0
                # Check all w < u
                # c(u, w) is stored in adj? No, adj is being built.
                # We can check adj[c][u] & ((1 << u) - 1) to find w < u with color c.
                # And adj[c][v] & ((1 << u) - 1) to find w < u with color c.
                # Intersection gives w such that both edges are color c.
                mask_u = adj[c][u] & ((1 << u) - 1)
                mask_v = adj[c][v] & ((1 << u) - 1)
                cost = (mask_u & mask_v).bit_count()
                
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
            
    return colors

def _sa_refine(n, colors, time_budget):
    """
    Simulated Annealing to minimize monochromatic triangles.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    adj = _build_adj(colors, n)
    tri_count = _count_mono_triangles(adj, n)
    
    best_tri = tri_count
    best_colors = colors[:]
    
    # Precompute edge endpoints for fast access
    # This is O(N^2) memory but N=50 is small (1225 edges)
    idx_to_edge = []
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge.append((i, j))
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while T > T_min and time.time() - start_time < time_budget and _time_left() > 5.0:
        # Perform a sweep or multiple random moves
        # For small N, random moves are fine
        for _ in range(50): # 50 moves per temperature step
            if time.time() - start_time > time_budget or _time_left() < 5.0:
                break
            
            ri = random.randint(0, m - 1)
            old_c = colors[ri]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            u, v = idx_to_edge[ri]
            
            # Calculate Delta E
            # Removed triangles: common neighbors of u and v in old_c
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            # Added triangles: common neighbors of u and v in new_c
            added = (adj[new_c][u] & adj[new_c][v]).bit_count()
            
            delta = added - removed
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept move
                colors[ri] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                tri_count += delta
                
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_colors = colors[:]
                    if tri_count == 0:
                        return best_colors, 0
        T *= cooling
        
    return best_colors, best_tri

def _random_init(n):
    m = n * (n - 1) // 2
    return [random.randint(0, 3) for _ in range(m)]

def construct():
    floor_colors = None
    floor_n = 0
    
    # We know n=32 is possible. Let's generate a valid one quickly as a fallback.
    # Using the Greenwood-Gleason construction for n=32.
    # This is a bit complex to code from scratch in the time available, 
    # but we can use a simple random restart for n=32 to find a valid one quickly, 
    # or just assume we can find n=50.
    # To be safe, we will try to find n=50. If it fails, we try n=40, then n=32.
    
    # Target n=50
    n_target = 50
    found = False
    best_colors = None
    
    # Try multiple initializations for n=50
    # 1. Greedy
    if _time_left() > 20.0:
        colors = _greedy_init(n_target)
        colors, tri = _sa_refine(n_target, colors, 30.0)
        if tri == 0:
            return {'n': n_target, 'colors': colors}
        best_colors = colors
        best_tri = tri
        
    # 2. Random
    if _time_left() > 20.0:
        colors = _random_init(n_target)
        colors, tri = _sa_refine(n_target, colors, 30.0)
        if tri == 0:
            return {'n': n_target, 'colors': colors}
        if best_colors is None or tri < best_tri:
            best_colors = colors
            best_tri = tri

    # 3. Another Random
    if _time_left() > 20.0:
        colors = _random_init(n_target)
        colors, tri = _sa_refine(n_target, colors, 30.0)
        if tri == 0:
            return {'n': n_target, 'colors': colors}
        if best_colors is None or tri < best_tri:
            best_colors = colors
            best_tri = tri

    # If n=50 failed, try n=40
    if _time_left() > 20.0:
        n_target = 40
        for _ in range(2):
            colors = _greedy_init(n_target)
            colors, tri = _sa_refine(n_target, colors, 20.0)
            if tri == 0:
                return {'n': n_target, 'colors': colors}

    # If n=40 failed, try n=32 (should be very easy)
    if _time_left() > 10.0:
        n_target = 32
        for _ in range(5):
            colors = _random_init(n_target)
            colors, tri = _sa_refine(n_target, colors, 5.0)
            if tri == 0:
                return {'n': n_target, 'colors': colors}
            # Greedy for 32
            colors = _greedy_init(n_target)
            colors, tri = _sa_refine(n_target, colors, 5.0)
            if tri == 0:
                return {'n': n_target, 'colors': colors}

    # Fallback: Return n=1 if all else fails (should not happen)
    return {'n': 1, 'colors': [0]}
