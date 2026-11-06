import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=27 first if time allows, otherwise n=26.
    # n=26 is known to exist. n=27 is the target for improvement.
    # We will allocate most time to n=26 to ensure success, then try n=27.
    
    # Strategy:
    # 1. Run SA for n=26 with 15s budget.
    # 2. If successful and time > 10s left, run SA for n=27 with remaining time.
    # 3. If n=27 fails, return n=26 result.
    # 4. If n=26 fails (unlikely), try n=25.
    
    result_26 = None
    
    # Try n=26
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 5
        # Give n=26 a good chunk of time, say 20s or half remaining
        t_26 = min(20, time_left)
        result_26 = solve_sa(26, time_limit=t_26)
        
    if result_26 is not None:
        # Try n=27 if we have time
        if time.time() < deadline - 5:
            time_left = deadline - time.time() - 5
            if time_left > 5:
                result_27 = solve_sa(27, time_limit=time_left)
                if result_27 is not None:
                    return result_27
        return result_26
        
    # Fallback to n=25 if 26 failed
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 5
        result_25 = solve_sa(25, time_limit=time_left)
        if result_25 is not None:
            return result_25
            
    # Final fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def solve_sa(n, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    # k[c][u][v] = number of common neighbors of u and v in color c
    # Use a flat array for speed: k[c][u*n + v]
    # Size: 5 * n * n
    k = [[0] * (n * n) for _ in range(5)]
    
    def calc_k(c):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                cnt = common.bit_count()
                k[c][u * n + v] = cnt
                k[c][v * n + u] = cnt
                
    # Initialize neighbors and k
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    for c in range(5):
        calc_k(c)
        
    # Calculate initial violations
    def total_violations():
        v = 0
        for c in range(5):
            for u in range(n):
                for v_idx in range(u + 1, n):
                    kc = k[c][u * n + v_idx]
                    if kc >= 2:
                        v += kc * (kc - 1) // 2
        return v
        
    current_violations = total_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    temperature = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Main SA loop
    while time.time() < deadline:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        i, j = edges[e_idx]
        
        delta = 0
        
        # Effect on old_color (removal)
        # For each w != i, j
        # If (j,w) is old_color, k(i,w) decreases by 1. Delta V += 1 - k(i,w)
        # If (i,w) is old_color, k(j,w) decreases by 1. Delta V += 1 - k(j,w)
        
        neigh_i_old = neighbors[old_color][i]
        neigh_j_old = neighbors[old_color][j]
        
        for w in range(n):
            if w == i or w == j:
                continue
            
            # Check if (j, w) is old_color
            # j is in N_old(i). We are removing (i,j).
            # N_old(i) loses j.
            # If w in N_old(i) AND w in N_old(j)? No.
            # k(i, w) = |N(i) \cap N(w)|.
            # N(i) loses j.
            # If j \in N(w), then j \in N(i) \cap N(w). So k(i,w) dec.
            # j \in N(w) iff (w,j) is old_color.
            if (neighbors[old_color][w] >> j) & 1:
                idx_k = i * n + w
                old_k = k[old_color][idx_k]
                delta += 1 - old_k
                k[old_color][idx_k] = old_k - 1
                k[old_color][w * n + i] = old_k - 1
                
            # k(j, w)
            # N(j) loses i.
            # If i \in N(w), then i \in N(j) \cap N(w). So k(j,w) dec.
            # i \in N(w) iff (w,i) is old_color.
            if (neighbors[old_color][w] >> i) & 1:
                idx_k = j * n + w
                old_k = k[old_color][idx_k]
                delta += 1 - old_k
                k[old_color][idx_k] = old_k - 1
                k[old_color][w * n + j] = old_k - 1
                
        # Effect on new_color (addition)
        # N_new(i) gains j. N_new(j) gains i.
        
        for w in range(n):
            if w == i or w == j:
                continue
            
            # If (j, w) is new_color
            # j \in N_new(w).
            # N_new(i) gains j.
            # If j \in N_new(w), then j \in N_new(i) \cap N_new(w). So k(i,w) inc.
            if (neighbors[new_color][w] >> j) & 1:
                idx_k = i * n + w
                old_k = k[new_color][idx_k]
                delta += old_k
                k[new_color][idx_k] = old_k + 1
                k[new_color][w * n + i] = old_k + 1
                
            # If (i, w) is new_color
            if (neighbors[new_color][w] >> i) & 1:
                idx_k = j * n + w
                old_k = k[new_color][idx_k]
                delta += old_k
                k[new_color][idx_k] = old_k + 1
                k[new_color][w * n + j] = old_k + 1
                
        # Acceptance
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations += delta
            colors[e_idx] = new_color
            
            # Update neighbors
            neighbors[old_color][i] &= ~(1 << j)
            neighbors[old_color][j] &= ~(1 << i)
            neighbors[new_color][i] |= (1 << j)
            neighbors[new_color][j] |= (1 << i)
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert k changes
            # It's faster to revert than recompute for small n?
            # Recomputing is O(n^2) for affected colors? No, only O(n) pairs involved.
            # But we stored the changes? No, we modified k in place.
            # We must revert k.
            
            # Revert old_color
            for w in range(n):
                if w == i or w == j:
                    continue
                # Recalc k(i,w)
                common = neighbors[old_color][i] & neighbors[old_color][w]
                cnt = common.bit_count()
                k[old_color][i * n + w] = cnt
                k[old_color][w * n + i] = cnt
                
                common = neighbors[old_color][j] & neighbors[old_color][w]
                cnt = common.bit_count()
                k[old_color][j * n + w] = cnt
                k[old_color][w * n + j] = cnt
                
            # Revert new_color
            for w in range(n):
                if w == i or w == j:
                    continue
                common = neighbors[new_color][i] & neighbors[new_color][w]
                cnt = common.bit_count()
                k[new_color][i * n + w] = cnt
                k[new_color][w * n + i] = cnt
                
                common = neighbors[new_color][j] & neighbors[new_color][w]
                cnt = common.bit_count()
                k[new_color][j * n + w] = cnt
                k[new_color][w * n + j] = cnt
            
            # Revert neighbors (not needed for k recompute, but needed for state consistency if we don't recompute neighbors? 
            # Wait, we didn't update neighbors in the 'accept' branch yet?
            # Yes, in 'accept' we update neighbors. In 'reject', we didn't.
            # So neighbors are still in the OLD state (before move).
            # So recompute k using CURRENT neighbors (which are old state) is correct.
            # But we modified k. So we overwrite k with recomputed values. Correct.
            
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
            
    return None
