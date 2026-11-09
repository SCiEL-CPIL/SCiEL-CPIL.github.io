import random
import time
import math

def construct():
    n = 256
    
    # Precompute all 4-APs and their positions
    all_aps = []
    pos_to_aps = [[] for _ in range(n)]
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for j in range(4):
                pos = ap[j]
                pos_to_aps[pos].append((idx, j))
                
    def count_mono(colors):
        count = 0
        for ap in all_aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    def calc_delta(colors, i):
        old_c = colors[i]
        new_c = 1 - old_c
        delta = 0
        for ap_idx, pos_in_ap in pos_to_aps[i]:
            ap = all_aps[ap_idx]
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            before_mono = (c0 == c1 == c2 == c3)
            
            if pos_in_ap == 0:
                after_mono = (new_c == c1 == c2 == c3)
            elif pos_in_ap == 1:
                after_mono = (c0 == new_c == c2 == c3)
            elif pos_in_ap == 2:
                after_mono = (c0 == c1 == new_c == c3)
            else:
                after_mono = (c0 == c1 == c2 == new_c)
                
            if before_mono and not after_mono:
                delta -= 1
            elif not before_mono and after_mono:
                delta += 1
        return delta

    # Heuristic Initialization: Greedy construction
    # We can try to minimize conflicts as we assign colors.
    # However, for 4-APs, a simple random start is often competitive.
    # Let's try a slightly smarter start: 
    # Start with random, then do a few quick greedy passes.
    
    def greedy_init():
        colors = [random.randint(0, 1) for _ in range(n)]
        # Do a few passes of local improvement
        for _ in range(10):
            improved = False
            for i in range(n):
                d_cost = calc_delta(colors, i)
                if d_cost < 0:
                    colors[i] = 1 - colors[i]
                    improved = True
            if not improved:
                break
        return colors

    colors = greedy_init()
    current_cost = count_mono(colors)
    best_cost = current_cost
    best_colors = colors.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.999 # Cooling rate
    
    # To make SA efficient, we need to handle the temperature decay and termination.
    # We will run until time runs out or T < T_min.
    
    while (time.time() - start_time) < time_limit:
        if T < T_min:
            break
            
        # Choose a random index to flip
        i = random.randint(0, n - 1)
        
        # Calculate delta
        d_cost = calc_delta(colors, i)
        
        # Acceptance criterion
        if d_cost < 0:
            # Always accept improvements
            colors[i] = 1 - colors[i]
            current_cost += d_cost
        else:
            # Accept worse solutions with probability exp(-d_cost / T)
            if random.random() < math.exp(-d_cost / T):
                colors[i] = 1 - colors[i]
                current_cost += d_cost
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors.copy()
            
        # Cool down
        T *= alpha
        
    # Final polish: run local search on the best solution found
    # This ensures we are at a local optimum of the best basin found
    final_colors = best_colors.copy()
    final_cost = best_cost
    
    # Simple hill climbing to ensure local optimality
    improved = True
    while improved:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            d_cost = calc_delta(final_colors, i)
            if d_cost < 0:
                final_colors[i] = 1 - final_colors[i]
                final_cost += d_cost
                improved = True
    
    if final_cost < best_cost:
        best_cost = final_cost
        best_colors = final_colors

    return best_colors
