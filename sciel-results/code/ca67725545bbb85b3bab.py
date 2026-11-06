import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin

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

def _local_search_bitset(n, colors, time_budget, seed=42):
    """
    Local search using bitsets for fast neighbor counting.
    """
    random.seed(seed)
    start_time = time.time()
    adj = _build_adj_bitsets(colors, n)
    tri_count = _count_mono_triangles(adj, n)
    
    if tri_count == 0:
        return colors, 0

    best_tri = tri_count
    best_colors = colors[:]
    
    # Precompute edge indices
    m = n * (n - 1) // 2
    # Map index to (u, v)
    # We need to be able to pick a random edge and know its endpoints and current color.
    # Instead of storing (u,v) for every index, we can generate random u,v and map to index.
    # Or just store the list.
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cool_rate = 0.9999
    
    while _time_left() > 5.0 and time.time() - start_time < time_budget:
        if tri_count == 0:
            break
            
        # Pick random edge
        idx = random.randint(0, m - 1)
        u, v = edges[idx]
        old_c = colors[idx]
        
        # Calculate delta for each new color
        best_new_c = old_c
        best_delta = 0 # We want to minimize delta, so if all are positive, we might still accept if T > 0
        
        # Current cost contribution of this edge to the total triangle count?
        # No, the formula is:
        # Delta = (Common neighbors in new_c) - (Common neighbors in old_c)
        # This is the change in the *number of triangles* if we flip.
        
        min_delta = float('inf')
        candidate_c = old_c
        
        for new_c in range(4):
            if new_c == old_c:
                continue
            # Number of common neighbors of u and v in color new_c
            # adj[new_c][u] & adj[new_c][v]
            common_new = (adj[new_c][u] & adj[new_c][v]).bit_count()
            common_old = (adj[old_c][u] & adj[old_c][v]).bit_count()
            
            delta = common_new - common_old
            
            if delta < min_delta:
                min_delta = delta
                candidate_c = new_c
                
        # If the best delta is negative, we definitely accept.
        # If positive, we might accept based on SA.
        # Note: delta is the change in the *total* triangle count.
        
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
    
    # Target n=50
    n_target = 50
    m_target = n_target * (n_target - 1) // 2
    
    # Initialize colors array
    colors = [0] * m_target
    
    # Copy K32 coloring
    colors_32 = floor_colors
    for i in range(32):
        for j in range(i + 1, 32):
            idx_new = _compute_index(n_target, i, j)
            idx_old = i * 31 - i * (i + 1) // 2 + (j - i - 1)
            colors[idx_new] = colors_32[idx_old]
    
    # Extend to n=50
    # For vertices 32 to 49, assign colors to edges (u, v) where u < v
    # Greedy: For each new vertex v, assign colors to (u, v) for u < v
    # to minimize the number of common neighbors in that color for future conflicts?
    # Actually, let's just assign randomly or greedily based on current adj.
    
    adj = [[0] * n_target for _ in range(4)]
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    random.seed(123)
    
    for v in range(32, n_target):
        # For each u < v, pick a color
        # Heuristic: Pick color c that minimizes the number of neighbors of u in color c
        # This balances the load.
        for u in range(v):
            best_c = 0
            min_count = float('inf')
            for c in range(4):
                # Count neighbors of u in color c that are < v (already assigned)
                # adj[c][u] & ((1 << u) - 1)
                count = (adj[c][u] & ((1 << u) - 1)).bit_count()
                if count < min_count:
                    min_count = count
                    best_c = c
            # Add some randomness
            if random.random() < 0.2:
                best_c = random.randint(0, 3)
                
            colors[_compute_index(n_target, u, v)] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
            
    # Now run local search
    # We need to pass the current colors and adj? 
    # _local_search_bitset rebuilds adj. That's fine.
    
    # Try multiple seeds for local search
    best_n = floor_n
    best_res_colors = floor_colors
    
    for seed in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        if _time_left() < 10.0:
            break
        budget = min(10.0, _time_left() - 5.0)
        if budget <= 0:
            break
        
        # Reset colors to the extended initial state?
        # The extension is deterministic given the seed 123.
        # So we start from the same extended state.
        # But local search modifies it.
        # We should re-extend or save the initial extended state.
        
        # Let's re-initialize the extended state for each seed to ensure diversity
        # in the local search start point?
        # Actually, the extension is fixed. The local search is stochastic.
        
        init_colors = [0] * m_target
        for i in range(32):
            for j in range(i + 1, 32):
                idx_new = _compute_index(n_target, i, j)
                idx_old = i * 31 - i * (i + 1) // 2 + (j - i - 1)
                init_colors[idx_new] = colors_32[idx_old]
                
        # Re-run greedy extension (deterministic part)
        adj_init = [[0] * n_target for _ in range(4)]
        idx = 0
        for i in range(32):
            for j in range(i + 1, 32):
                c = init_colors[idx]
            adj_init[c][i] |= (1 << j)
            adj_init[c][j] |= (1 << i)
            idx += 1
            
        random.seed(123 + seed) # Vary the random part of extension slightly?
        # No, let's keep extension fixed and vary local search.
        # But if extension is fixed, local search starts from same point.
        # Better to vary the extension slightly.
        
        for v in range(32, n_target):
            for u in range(v):
                best_c = 0
                min_count = float('inf')
                for c in range(4):
                    count = (adj_init[c][u] & ((1 << u) - 1)).bit_count()
                    if count < min_count:
                        min_count = count
                        best_c = c
                if random.random() < 0.3:
                    best_c = random.randint(0, 3)
                    
                init_colors[_compute_index(n_target, u, v)] = best_c
                adj_init[best_c][u] |= (1 << v)
                adj_init[best_c][v] |= (1 << u)
                
        # Run local search
        res_colors, tri = _local_search_bitset(n_target, init_colors, time_budget=budget, seed=seed)
        
        if tri == 0:
            return {'n': n_target, 'colors': res_colors}
            
    # Fallback to K32
    return {'n': floor_n, 'colors': floor_colors}
