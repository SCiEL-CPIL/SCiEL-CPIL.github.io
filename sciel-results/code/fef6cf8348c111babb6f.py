import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Known algebraic construction for n=26 (Lazebnik-Woldar)
    # This is deterministic and instant. If it works, we get score 26/11 ≈ 2.36
    # Let's try to implement the known construction for R_5(C4) > 26.
    # The construction uses vertices as pairs (x, y) in Z_5 x Z_5 plus one "infinity" point?
    # Actually, the standard result R_5(C4) > 25 comes from the affine plane of order 5.
    # The result R_5(C4) > 26 is from Lazebnik and Woldar (1995).
    # Let's see if we can find a valid coloring for n=26 quickly using local search.
    # If not, we fall back to n=25 (which is easier) or lower.
    
    # Strategy:
    # 1. Try to find a valid coloring for n=26 using fast local search with incremental updates.
    # 2. If that fails within a time budget, try n=25.
    # 3. If that fails, try n=24, etc.
    
    target_ns = [26, 25, 24, 23, 22]
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
        # Allocate more time for larger n
        time_limit = 15 if n == 26 else 10
        result = local_search_fast(n, time_limit)
        if result is not None:
            return result
            
    # Fallback to seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search_fast(n, time_limit=10):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge index for O(1) lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build adjacency masks for each color
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        # Remove from old_color
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        # Add to new_color
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Calculate violations
    # A violation is a pair of vertices (u, v) and a color c such that they have >1 common neighbor in color c.
    # We count the number of C4s.
    def calc_violations():
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations
        
    current_violations = calc_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    # Local search loop
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Apply change
        colors[e_idx] = new_color
        update_neighbors(e_idx, old_color, new_color)
        
        # Calculate new violations efficiently?
        # Changing edge (i,j) from c1 to c2 affects:
        # 1. Violations in c1 involving i or j
        # 2. Violations in c2 involving i or j
        # But calculating the full violation count is O(5 * n^2 * bit_count).
        # For n=26, n^2 = 676. 5*676 = 3380 checks. bit_count is fast.
        # It's likely fast enough to just recalculate everything.
        
        new_violations = calc_violations()
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            colors[e_idx] = old_color
            update_neighbors(e_idx, new_color, old_color)
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None
