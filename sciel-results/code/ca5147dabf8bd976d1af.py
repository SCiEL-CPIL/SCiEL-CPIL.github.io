import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

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
    return colors

def _compute_index(n, i, j):
    """Calculate the index of edge (i, j) in the lexicographic list of edges for K_n."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj_bitsets(colors, n):
    """Build adjacency as list of 4 lists of bitsets (integers)."""
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
    """Count total number of monochromatic triangles."""
    count = 0
    for c in range(4):
        for i in range(n):
            # Count triangles i-j-k where i<j<k
            # Triangles involving i as the smallest vertex
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                # Common neighbors of i and j, strictly greater than j
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _greedy_init(n, seed=42):
    """
    Initialize coloring by adding vertices one by one.
    For each new vertex v, assign colors to edges (u, v) for u < v
    to minimize the number of monochromatic triangles created.
    """
    random.seed(seed)
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                # If we color (u, v) with c, we create a triangle for every w < u 
                # such that (u, w) is color c AND (v, w) is color c? 
                # No, (v, w) is not assigned yet if w < u < v? 
                # Wait, we are adding v. Edges (w, v) for w < v are being assigned now.
                # When we assign (u, v) with color c, it forms a triangle with any w < u 
                # IF (w, u) is color c AND (w, v) is color c.
                # But (w, v) is not assigned yet for w < u? 
                # Actually, we iterate u from 0 to v-1.
                # When we are at u, we have already assigned (w, v) for all w < u.
                # So we can check (w, v) colors.
                
                # Let's re-evaluate the cost.
                # Cost of coloring (u, v) with c:
                # 1. Triangles (u, v, w) where w < u.
                #    These exist if (u, w) is c AND (v, w) is c.
                #    We have assigned (v, w) for all w < u.
                #    So we need to count w < u such that adj[c][u] has bit w AND adj[c][v] has bit w.
                #    Note: adj[c][v] currently only has bits for w < u because we are processing u in increasing order.
                
                common = (adj[c][u] & adj[c][v]).bit_count()
                
                # 2. Potential future triangles? 
                #    If we just minimize immediate triangles, it's a greedy heuristic.
                
                if common < min_cost:
                    min_cost = common
                    best_c = c
            
            # Add some randomness to escape local minima early? 
            # Or just pick best. Let's stick to best for now, SA will fix later.
            # If we want diversity, we can randomize among top-k.
            
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
            
    return colors, adj

def _local_search(n, time_budget, seed=42):
    """
    Simulated Annealing to minimize monochromatic triangles.
    Returns (colors, tri_count).
    """
    random.seed(seed)
    start_time = time.time()
    
    # Initialize
    colors, adj = _greedy_init(n, seed)
    tri_count = _count_mono_triangles(adj, n)
    
    if tri_count == 0:
        return colors, 0

    best_tri = tri_count
    best_colors = colors[:]
    
    m = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    T = 5.0
    T_min = 0.01
    cool_rate = 0.9995
    
    while _time_left() > 2.0 and time.time() - start_time < time_budget:
        if tri_count == 0:
            return colors, 0
            
        # Pick random edge
        idx = random.randint(0, m - 1)
        u, v = edges[idx]
        old_c = colors[idx]
        
        # Find best new color
        min_delta = float('inf')
        candidate_c = old_c
        
        # Common neighbors in old color
        common_old = (adj[old_c][u] & adj[old_c][v]).bit_count()
        
        for new_c in range(4):
            if new_c == old_c:
                continue
            common_new = (adj[new_c][u] & adj[new_c][v]).bit_count()
            delta = common_new - common_old
            
            if delta < min_delta:
                min_delta = delta
                candidate_c = new_c
                
        # Acceptance probability
        if min_delta < 0 or (T > T_min and random.random() < math.exp(-min_delta / T)):
            # Apply move
            colors[idx] = candidate_c
            
            # Update adj
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[candidate_c][u] |= (1 << v)
            adj[candidate_c][v] |= (1 << u)
            
            tri_count += min_delta
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = colors[:]
                if tri_count == 0:
                    return colors, 0
        
        T *= cool_rate

    return best_colors, best_tri

def construct():
    # Guaranteed floor: K32 valid coloring.
    floor_colors = _seed_32()
    floor_n = 32
    
    # Try to find valid colorings for n > 32
    # We prioritize n=50 (known record) then decrease.
    # We allocate time based on remaining budget.
    
    # To ensure we don't timeout, we iterate n from 50 down to 33.
    # We give more time to higher n.
    
    # Estimate time per n.
    # Let's try n=50 first with a significant chunk of time.
    # If it fails, try 49, etc.
    
    # Strategy: 
    # 1. Try n=50 for ~80s.
    # 2. If fail, try n=49 for ~20s.
    # 3. If fail, try n=48 for ~5s.
    # 4. Return floor.
    
    # Actually, it's better to try a few restarts for n=50 if possible.
    # But one long run with SA might be enough if the init is good.
    
    # Let's structure the time allocation:
    # Total available ~110s.
    # n=50: 80s
    # n=49: 20s
    # n=48: 5s
    # n=47: 5s
    
    target_ns = [50, 49, 48, 47, 46, 45]
    time_allocations = [80.0, 20.0, 5.0, 5.0, 5.0, 5.0]
    
    for n, budget in zip(target_ns, time_allocations):
        if _time_left() < 10.0:
            break
            
        # Adjust budget to not exceed deadline
        actual_budget = min(budget, _time_left() - 5.0)
        if actual_budget <= 0:
            break
            
        # Run local search
        # We can run multiple restarts if time permits?
        # _local_search does one run. 
        # Let's run it once. If it finds 0, we are done.
        
        colors, tri = _local_search(n, actual_budget, seed=n)
        
        if tri == 0:
            return {'n': n, 'colors': colors}
            
    # Fallback to K32
    return {'n': floor_n, 'colors': floor_colors}
