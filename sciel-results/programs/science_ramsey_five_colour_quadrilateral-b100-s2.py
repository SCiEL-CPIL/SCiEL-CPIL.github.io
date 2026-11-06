import time
import random
import math

def construct():
    deadline = time.time() + 115
    
    # Try 26 first, then 25, etc.
    # If we find 26, return it.
    
    best_n = 0
    best_colors = []
    
    for n in [26, 25, 24, 23, 22]:
        if time.time() > deadline - 5:
            break
            
        # Time budget
        if n == 26:
            t_limit = 50
        elif n == 25:
            t_limit = 30
        else:
            t_limit = 10
            
        actual_limit = min(deadline - time.time() - 2, t_limit)
        if actual_limit < 1:
            continue
            
        success, colors = _fast_sa(n, time.time() + actual_limit)
        if success:
            best_n = n
            best_colors = colors
            break
            
    if best_n == 0:
        # Fallback to a small valid coloring if all else fails
        # n=11 is usually easy
        success, colors = _fast_sa(11, time.time() + 5)
        if success:
            best_n = 11
            best_colors = colors
        else:
            # Hardcode a valid coloring for n=11? 
            # Or just return n=1 with empty colors? No, must be valid.
            # For n=1, no edges, valid.
            best_n = 1
            best_colors = []

    return {'n': best_n, 'colors': best_colors}

def _fast_sa(n, deadline):
    m = n * (n - 1) // 2
    if m == 0:
        return True, []
        
    # colors[i] is color of edge i
    # edge_u[i], edge_v[i]
    edge_u = [0] * m
    edge_v = [0] * m
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # Initialize colors randomly
    colors = [random.randint(0, 4) for _ in range(m)]
    
    # common_count[c][u][v] is number of common neighbors of u and v in color c
    # Use a 3D list or dict. n=26 is small, 26*26*5 is small.
    common_count = [[[0]*n for _ in range(n)] for _ in range(5)]
    
    # Initialize common_count
    for c in range(5):
        for u in range(n):
            for v in range(u+1, n):
                count = 0
                for w in range(n):
                    if w == u or w == v: continue
                    if colors[pair_to_idx[(u,w)]] == c and colors[pair_to_idx[(v,w)]] == c:
                        count += 1
                common_count[c][u][v] = count
                common_count[c][v][u] = count
                
    # Count initial violations
    violations = 0
    for c in range(5):
        for u in range(n):
            for v in range(u+1, n):
                if common_count[c][u][v] >= 2:
                    violations += 1
                    
    best_violations = violations
    best_colors = colors[:]
    
    if violations == 0:
        return True, colors
        
    T = 1.0
    alpha = 0.999
    min_T = 0.01
    start_time = time.time()
    
    while time.time() < deadline:
        k = random.randint(0, m - 1)
        u = edge_u[k]
        v = edge_v[k]
        old_c = colors[k]
        new_c = random.randint(0, 4)
        
        if new_c == old_c:
            continue
            
        delta = 0
        
        # Update common_count and violations for affected pairs
        # Affected pairs: (u, w) and (v, w) for all w != u, v
        
        # We need to store the old values to revert if needed
        # Or calculate delta directly
        
        # Pairs (u, w) and (v, w)
        # For each w:
        # 1. Check color of (w, v). If old_c, common_count[old_c][u][w] decreases.
        # 2. Check color of (w, v). If new_c, common_count[new_c][u][w] increases.
        # 3. Check color of (w, u). If old_c, common_count[old_c][v][w] decreases.
        # 4. Check color of (w, u). If new_c, common_count[new_c][v][w] increases.
        
        # We must be careful: (u,v) edge is changing, so it doesn't contribute to common neighbors of (u,w) via v?
        # Common neighbors of u and w are z such that (u,z) and (w,z) are same color.
        # If z=v, we need (u,v) and (w,v) to be same color.
        # So yes, the status of (u,v) matters for pairs (u,w) and (v,w).
        
        # Let's trace carefully.
        # Change (u,v) from old_c to new_c.
        
        # For each w in range(n):
        #   if w == u or w == v: continue
          
        #   # Pair (u, w)
        #   # Common neighbors in old_c:
        #   # v was a common neighbor if colors[(v,w)] == old_c.
        #   # Now (u,v) is new_c, so v is NOT a common neighbor in old_c anymore.
        #   # So common_count[old_c][u][w] decreases by 1 if colors[(v,w)] == old_c.
        #   # v was NOT a common neighbor in new_c (since (u,v) was old_c).
        #   # Now (u,v) is new_c, so v IS a common neighbor in new_c if colors[(v,w)] == new_c.
        #   # So common_count[new_c][u][w] increases by 1 if colors[(v,w)] == new_c.
          
        #   # Pair (v, w)
        #   # Similarly:
        #   # common_count[old_c][v][w] decreases by 1 if colors[(u,w)] == old_c.
        #   # common_count[new_c][v][w] increases by 1 if colors[(u,w)] == new_c.
        
        # We need to update `violations` based on these changes.
        # A change in count from 1 to 2 increases violations by 1.
        # A change in count from 2 to 1 decreases violations by 1.
        
        # Let's apply changes tentatively.
        
        revert_info = []
        
        for w in range(n):
            if w == u or w == w == v: # Typo in thought, should be w == u or w == v
                continue
            if w == u or w == v:
                continue
                
            # Pair (u, w)
            # Check (v, w) color
            c_vw = colors[pair_to_idx[(v, w)]]
            if c_vw == old_c:
                # common_count[old_c][u][w] decreases
                old_val = common_count[old_c][u][w]
                new_val = old_val - 1
                common_count[old_c][u][w] = new_val
                common_count[old_c][w][u] = new_val
                if old_val == 2:
                    delta -= 1
                revert_info.append(('dec', old_c, u, w, old_val))
                
            if c_vw == new_c:
                # common_count[new_c][u][w] increases
                old_val = common_count[new_c][u][w]
                new_val = old_val + 1
                common_count[new_c][u][w] = new_val
                common_count[new_c][w][u] = new_val
                if old_val == 1:
                    delta += 1
                revert_info.append(('inc', new_c, u, w, old_val))
                
            # Pair (v, w)
            c_uw = colors[pair_to_idx[(u, w)]]
            if c_uw == old_c:
                # common_count[old_c][v][w] decreases
                old_val = common_count[old_c][v][w]
                new_val = old_val - 1
                common_count[old_c][v][w] = new_val
                common_count[old_c][w][v] = new_val
                if old_val == 2:
                    delta -= 1
                revert_info.append(('dec', old_c, v, w, old_val))
                
            if c_uw == new_c:
                # common_count[new_c][v][w] increases
                old_val = common_count[new_c][v][w]
                new_val = old_val + 1
                common_count[new_c][v][w] = new_val
                common_count[new_c][w][v] = new_val
                if old_val == 1:
                    delta += 1
                revert_info.append(('inc', new_c, v, w, old_val))
                
        # Also, the pair (u, v) itself?
        # Common neighbors of u and v?
        # The edge (u,v) itself is not a common neighbor.
        # But does changing (u,v) affect the common neighbor count of (u,v)?
        # No, common neighbors are other vertices w.
        # So no change to common_count[?][u][v].
        
        violations += delta
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            colors[k] = new_c
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                if best_violations == 0:
                    return True, best_colors
        else:
            # Revert
            colors[k] = old_c
            for op, c, x, y, old_val in revert_info:
                if op == 'dec':
                    common_count[c][x][y] = old_val
                    common_count[c][y][x] = old_val
                else:
                    common_count[c][x][y] = old_val
                    common_count[c][y][x] = old_val
            violations -= delta
            
        T *= alpha
        if T < min_T:
            T = min_T
            
    if best_violations == 0:
        return True, best_colors
    return False, best_colors
