import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    
    best_result = None
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        result = local_search(n, time_limit=15)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        # Fallback: Return a trivially valid small n
        best_result = {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
        
    return best_result

def local_search(n, time_limit=15):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Precompute edge indices
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize neighbor masks
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    def calc_violations(neighbors):
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    current_violations = calc_violations(neighbors)
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    
    # Simulated Annealing / Hill Climbing
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Calculate delta in violations
        delta = 0
        
        # 1. Remove from old_color
        # Update violations for c = old_color
        c = old_color
        # Pairs involving i
        # For each v in N_c(j) \ {i}, k_iv increases/decreases?
        # We are REMOVING j from N_c(i).
        # So for v in N_c(j) \ {i}, j is a common neighbor of i and v? No.
        # j is in N_c(i) currently.
        # If v is in N_c(j), then j is a neighbor of v.
        # Since j is also a neighbor of i, j is a common neighbor of i and v.
        # When we remove j from N_c(i), j is no longer a common neighbor of i and v.
        # So k_iv decreases by 1.
        # Change in binom(k,2) when k -> k-1 is -(k-1).
        
        mask_j = neighbors[c][j]
        # Iterate over neighbors of j in color c
        v = 0
        m = mask_j
        while m:
            if m & 1:
                if v != i:
                    # k is current count of common neighbors of i and v
                    # This includes j?
                    # Yes, because j is in N_c(i) and j is in N_c(v) (since v is in N_c(j) implies j in N_c(v)).
                    # So current k includes j.
                    # We are removing j, so new k is k-1.
                    # We need to subtract (k-1).
                    k = (neighbors[c][i] & neighbors[c][v]).bit_count()
                    delta -= (k - 1)
            m >>= 1
            v += 1
            
        # Pairs involving j
        # We are removing i from N_c(j).
        # For each v in N_c(i) \ {j}, i is a common neighbor of j and v.
        # When we remove i from N_c(j), k_jv decreases by 1.
        # Change is -(k_jv - 1).
        
        mask_i = neighbors[c][i]
        v = 0
        m = mask_i
        while m:
            if m & 1:
                if v != j:
                    k = (neighbors[c][j] & neighbors[c][v]).bit_count()
                    delta -= (k - 1)
            m >>= 1
            v += 1
            
        # Apply removal to masks
        neighbors[c][i] &= ~(1 << j)
        neighbors[c][j] &= ~(1 << i)
        
        # 2. Add to new_color
        c = new_color
        # We are adding j to N_c(i).
        # For each v in N_c(j) \ {i}, j becomes a common neighbor of i and v.
        # k_iv increases by 1.
        # Change in binom(k,2) when k -> k+1 is k.
        # Note: k is the count BEFORE adding j.
        
        mask_j = neighbors[c][j]
        v = 0
        m = mask_j
        while m:
            if m & 1:
                if v != i:
                    k = (neighbors[c][i] & neighbors[c][v]).bit_count()
                    delta += k
            m >>= 1
            v += 1
            
        # We are adding i to N_c(j).
        # For each v in N_c(i) \ {j}, i becomes a common neighbor of j and v.
        # k_jv increases by 1.
        # Change is k.
        
        mask_i = neighbors[c][i]
        v = 0
        m = mask_i
        while m:
            if m & 1:
                if v != j:
                    k = (neighbors[c][j] & neighbors[c][v]).bit_count()
                    delta += k
            m >>= 1
            v += 1
            
        # Apply addition to masks
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
        # Update colors array
        colors[e_idx] = new_color
        
        new_violations = current_violations + delta
        
        # Accept if better, or with probability exp(-delta / temperature)
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            # Undo mask changes
            # Revert new_color addition
            neighbors[c][i] &= ~(1 << j)
            neighbors[c][j] &= ~(1 << i)
            
            # Revert old_color removal (i.e., add back)
            c = old_color
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
            colors[e_idx] = old_color
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None
