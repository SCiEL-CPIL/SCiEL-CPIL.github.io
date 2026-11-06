import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Target n values to try, in descending order
    # We prioritize n=26 as it is the known maximum
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    best_result = {'n': 11, 'colors': [0] * 55} # Fallback
    
    for n in target_ns:
        if time.time() > deadline:
            break
        
        # Allocate time: more for larger n
        if n == 26:
            time_limit = 90
        elif n == 25:
            time_limit = 15
        elif n == 24:
            time_limit = 10
        else:
            time_limit = 5
            
        # Ensure we don't exceed deadline
        current_time = time.time()
        if current_time + time_limit > deadline:
            time_limit = deadline - current_time
            if time_limit < 1:
                continue

        result = local_search(n, time_limit)
        if result is not None:
            best_result = result
            # If we found a solution for a large n, we can stop 
            # unless we want to try for even larger? 
            # The task says maximize n. 26 is the known max. 
            # If we find 26, we stop.
            if n == 26:
                break
            # If we find 25, should we try 26 again with remaining time?
            # The loop goes 26, 25... so if 26 failed, we try 25.
            # If 25 succeeds, we return 25.
            # This is correct.
            
    return best_result

def local_search(n, time_limit):
    """
    Attempts to find a valid 5-coloring of K_n with no C4 in any color class.
    Uses Simulated Annealing.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_violations(colors):
        # neighbors[c][v] is a bitmask of neighbors of v in color c
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
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    best_colors = None
    best_violations = float('inf')
    
    start_time = time.time()
    
    # Multiple restarts
    while time.time() - start_time < time_limit:
        # Initialize random coloring
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        current_violations = get_violations(colors)
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            if best_violations == 0:
                return best_colors
        
        # Simulated Annealing parameters
        temp = 10.0
        cooling = 0.995
        max_steps = 5000 # Steps per restart
        
        for _ in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Try all possible new colors
            best_new_color = old_color
            best_new_violations = current_violations
            
            for new_color in range(5):
                if new_color == old_color:
                    continue
                colors[e_idx] = new_color
                new_violations = get_violations(colors)
                if new_violations < best_new_violations:
                    best_new_violations = new_violations
                    best_new_color = new_color
            
            colors[e_idx] = best_new_color
            
            if best_new_violations < current_violations:
                # Accept improvement
                current_violations = best_new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors
            else:
                # Accept worse move with probability exp(-delta / temp)
                delta = best_new_violations - current_violations
                if delta > 0:
                    # Avoid overflow in exp
                    if delta / temp < 50:
                        prob = pow(2.718281828, -delta / temp)
                    else:
                        prob = 0.0
                    if random.random() < prob:
                        current_violations = best_new_violations
                    else:
                        # Reject move
                        colors[e_idx] = old_color
            
            temp *= cooling
            if temp < 0.01:
                break
                
    return best_colors
