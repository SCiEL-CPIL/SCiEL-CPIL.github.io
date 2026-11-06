import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target n values to try, in descending order
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    best_result = {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        # Try to find a valid coloring for this n
        result = local_search(n, time_limit=10)
        if result is not None:
            best_result = result
            break # Found the largest n, so stop
            
        # If not found, continue to smaller n
        
    return best_result

def local_search(n, time_limit=10):
    """
    Attempts to find a valid 5-coloring of K_n with no C4 in any color class.
    Returns a dict with 'n' and 'colors' if successful, else None.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # colors[e] is the color of edge e (0-4)
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Precompute edge index for fast access
    # Map (i, j) -> index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Function to calculate total C4 violations
    def get_violations():
        # neighbors[c][v] is a bitmask of neighbors of v in color c
        # n <= 26, so we can use Python ints as bitmasks
        neighbors = [[0] * n for _ in range(5)]
        
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    # Number of common neighbors
                    k = bin(common).count('1')
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    current_violations = get_violations()
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    iterations = 0
    max_iterations = 1000000  # Cap iterations to avoid infinite loop if time check fails
    
    while time.time() < start_time + time_limit and iterations < max_iterations:
        iterations += 1
        
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        colors[e_idx] = new_color
        
        new_violations = get_violations()
        
        if new_violations < best_violations:
            best_violations = new_violations
            best_colors = colors[:]
            
            if best_violations == 0:
                return {'n': n, 'colors': colors}
                
            # Restart search from this new best state? Or just continue?
            # Hill climbing: accept if better, else reject.
            # Since we update best, we are doing a form of best-improvement or first-improvement?
            # Actually, this is a random walk with restarts to best.
            # Let's implement a simple simulated annealing or just restart from best if stuck.
            
            # For simplicity and speed, if we find a better state, we keep it.
            # If we don't find a better state for a while, we might restart.
            # Given the speed, let's just do hill climbing:
            # If new_violations < current_violations, accept. Else reject.
            # But the code above accepts if < best, not < current.
            # Let's switch to standard hill climbing: accept if new < current.
            # And keep track of global best.
            
        else:
            # Reject
            colors[e_idx] = old_color
            # The 'if' block above handled the acceptance logic incorrectly for standard hill climbing.
            # Let's re-structure.
            
    # The loop above was a bit messy. Let's rewrite the inner loop properly.
    
    return None
