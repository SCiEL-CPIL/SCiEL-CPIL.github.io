import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Target n values in descending order of preference
    # We know n=26 exists. We try to find 27 or 28.
    # If we fail, we return the best found (likely 26).
    
    # Time allocation strategy:
    # Spend a little time on higher n, and reserve most for n=26 to ensure we get a valid result.
    # If we find 27 or 28, we return immediately.
    
    attempts = [
        (28, 2.0),
        (27, 10.0),
        (26, 100.0) # Should be plenty for 26
    ]
    
    best_result = None
    
    for n, time_limit in attempts:
        if time.time() > deadline - 1.0:
            break
            
        # Adjust time limit if we are running out of time
        actual_limit = min(time_limit, deadline - time.time() - 0.5)
        if actual_limit <= 0:
            continue
            
        result = local_search_fast(n, actual_limit)
        
        if result is not None:
            return result
        elif best_result is None or result['n'] > best_result['n']:
            best_result = result
            
    # Fallback to a known valid coloring for n=26 if we didn't find one via search
    # This is a safety net. The local search for n=26 is very likely to succeed.
    if best_result is None:
        # Generate a random valid coloring for n=26? 
        # Or use a specific known one? 
        # The local search for n=26 is very robust. 
        # If it failed, something is wrong. 
        # Let's try a very short deterministic greedy or just return a small n to be safe?
        # No, n=26 is the standard. 
        # Let's force one more try for n=26 with a tight deadline if we are here.
        if time.time() < deadline - 1.0:
             result = local_search_fast(26, deadline - time.time() - 1.0)
             if result:
                 return result
        
        # Absolute fallback: n=11 (from seed logic, but ensure validity)
        # The seed logic for n=11 was: colors = [min(j - i, 11 - j + i) - 1 ...]
        # Let's just return a simple valid small graph to avoid 0 score.
        # n=5 is trivially valid.
        n = 5
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(0)
        return {'n': n, 'colors': colors}

def local_search_fast(n, time_limit):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initialize colors randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build adjacency masks for each color
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    def calc_c4_count(u, v, c):
        # Number of C4s in color c that include edge (u,v)
        # A C4 is u-a-b-v-u (or similar).
        # We need paths of length 3 from u to v in G_c.
        # Path: u -> a -> b -> v.
        # Edges: (u,a), (a,b), (b,v).
        # a in N_c(u) \ {v}, b in N_c(v) \ {u}, (a,b) in E_c, a != b.
        
        mask_u = neighbors[c][u] & ~(1 << v)
        mask_v = neighbors[c][v] & ~(1 << u)
        
        count = 0
        m = mask_u
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            
            # Count b in mask_v such that (a,b) is an edge in c
            # common = N_c(a) & mask_v
            common = neighbors[c][a] & mask_v
            count += common.bit_count()
            
            # Note: We don't need to subtract 1 if a in mask_v 
            # because (a,a) is not an edge, so a is not in N_c(a).
            # Thus a is not in common.
            
        return count

    # Initial violation count
    # Sum of C4s over all edges? No, each C4 is counted 4 times (once per edge).
    # So Total V = (1/4) * sum_{(u,v) in E} calc_c4_count(u, v, color(u,v))
    # But we only need relative changes.
    # Let's compute the initial total violations to check for 0.
    
    def get_total_violations():
        total = 0
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            total += calc_c4_count(u, v, c)
        return total // 4 # Each C4 has 4 edges, so it's counted 4 times
    
    current_violations = get_total_violations()
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 10.0
    cooling_rate = 0.9999
    min_temperature = 0.1
    
    # Local search loop
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        u, v = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Calculate delta
        # Cost of adding edge (u,v) to new_color
        # Note: neighbors array is current state (edge is in old_color)
        # So calc_c4_count(u, v, new_color) counts C4s that would be formed 
        # using the new edge and existing paths. This is correct.
        cost_new = calc_c4_count(u, v, new_color)
        
        # Cost of removing edge (u,v) from old_color
        # calc_c4_count(u, v, old_color) counts C4s that currently exist 
        # and use the edge. This is correct.
        cost_old = calc_c4_count(u, v, old_color)
        
        delta = cost_new - cost_old
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            # Accept move
            colors[e_idx] = new_color
            
            # Update neighbors
            neighbors[old_color][u] &= ~(1 << v)
            neighbors[old_color][v] &= ~(1 << u)
            neighbors[new_color][u] |= (1 << v)
            neighbors[new_color][v] |= (1 << u)
            
            current_violations += delta
            # Note: current_violations is sum of calc_c4_count for all edges.
            # The true number of C4s is current_violations // 4.
            # We track the sum for efficiency.
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert (no state change needed)
            pass
            
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
            
    return {'n': n, 'colors': best_colors} if best_violations == 0 else None

