import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 115
    
    # Precompute all 4-APs
    aps = []
    aps_at = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a+d, a+2*d, a+3*d)
            aps.append(idxs)
            for i in range(4):
                cur = idxs[i]
                others = [idxs[j] for j in range(4) if j != i]
                aps_at[cur].append(others)
    
    # Convert to numpy arrays for faster computation
    aps_np = np.array(aps, dtype=np.int32)
    aps_at_np = [np.array(lst, dtype=np.int32) if lst else np.empty((0,3), dtype=np.int32) for lst in aps_at]
    
    def count_mono(cols):
        cols_np = np.array(cols, dtype=np.int8)
        # Vectorized count: check if all 4 elements in each AP are equal
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def delta_for_flip(cols, i):
        """Calculate cost delta if we flip position i."""
        c_i = cols[i]
        others_arr = aps_at_np[i]
        if len(others_arr) == 0:
            return 0
        cols_np = np.array(cols, dtype=np.int8)
        c_others = cols_np[others_arr]
        # Check if all 3 others are the same color
        all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
        n_same = int(np.sum(all_same))
        # For those where all 3 are same, check if they match c_i
        if n_same > 0:
            matching = (c_others[all_same, 0] == c_i).sum()
            return -(int(matching)) + (n_same - int(matching))
        return 0
    
    def greedy_polish(colors, current_cost):
        """Try all single flips and apply any that reduce cost."""
        improved = True
        while improved:
            improved = False
            for i in range(N):
                d = delta_for_flip(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    current_cost += d
                    improved = True
        return colors, current_cost
    
    def pair_swap_polish(colors, current_cost):
        """Try all pair swaps and apply any that reduce cost."""
        improved = True
        while improved:
            improved = False
            for i in range(N):
                if not improved and time.time() > deadline - 10:
                    break
                for j in range(i+1, N):
                    if colors[i] == colors[j]:
                        continue
                    # Try swapping i and j
                    colors[i], colors[j] = colors[j], colors[i]
                    new_cost = count_mono(colors)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                    else:
                        colors[i], colors[j] = colors[j], colors[i]
                    if not improved and time.time() > deadline - 10:
                        break
        return colors, current_cost
    
    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit):
        colors = initial_colors[:]
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            i = random.randint(0, N - 1)
            delta = delta_for_flip(colors, i)
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            elif delta == 0:
                colors[i] = 1 - colors[i]
            else:
                if T > 0:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                
                if random.random() < prob:
                    colors[i] = 1 - colors[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_colors, best_cost
    
    def generate_good_initial():
        """Generate a good initial coloring using known patterns."""
        # Try several pattern-based initializations
        patterns = []
        
        # Pattern 1: Based on binary representation
        p1 = [(i & 1) ^ ((i >> 1) & 1) for i in range(N)]
        patterns.append(p1)
        
        # Pattern 2: Block pattern with varying sizes
        p2 = []
        block_size = 4
        for i in range(N):
            block_idx = i // block_size
            p2.append(block_idx % 2)
        patterns.append(p2)
        
        # Pattern 3: Random with bias
        p3 = [random.randint(0, 1) for _ in range(N)]
        patterns.append(p3)
        
        # Pattern 4: Striped pattern
        p4 = [(i // 4) % 2 for i in range(N)]
        patterns.append(p4)
        
        # Evaluate all patterns and return the best
        best_pattern = None
        best_cost = float('inf')
        for p in patterns:
            cost = count_mono(p)
            if cost < best_cost:
                best_cost = cost
                best_pattern = p[:]
        
        return best_pattern, best_cost
    
    # Global best
    global_best_colors = None
    global_best_cost = float('inf')
    
    # Start with a good initial pattern
    initial_pattern, initial_cost = generate_good_initial()
    if initial_cost < global_best_cost:
        global_best_cost = initial_cost
        global_best_colors = initial_pattern[:]
    
    # Multiple restarts with different strategies
    num_restarts = 12
    time_per_restart = 105.0 / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Different initial colorings for diversity
        if restart == 0:
            initial = initial_pattern[:]
        elif restart == 1:
            # Perturb previous best significantly
            if global_best_colors is not None:
                initial = global_best_colors[:]
                for _ in range(N // 2):
                    idx = random.randint(0, N - 1)
                    initial[idx] = 1 - initial[idx]
            else:
                initial = [random.randint(0, 1) for _ in range(N)]
        elif restart == 2:
            # Alternating with noise
            initial = [i % 2 for i in range(N)]
            for _ in range(N // 8):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        elif restart == 3:
            # Block pattern
            initial = [(i // 8) % 2 for i in range(N)]
            for _ in range(N // 8):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        elif restart == 4:
            # Random
            initial = [random.randint(0, 1) for _ in range(N)]
        elif restart == 5:
            # Binary pattern
            initial = [(i & 1) ^ ((i >> 1) & 1) for i in range(N)]
            for _ in range(N // 8):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        else:
            # Perturb previous best moderately
            if global_best_colors is not None:
                initial = global_best_colors[:]
                for _ in range(N // 6):
                    idx = random.randint(0, N - 1)
                    initial[idx] = 1 - initial[idx]
            else:
                initial = [random.randint(0, 1) for _ in range(N)]
        
        # Simulated annealing with varying parameters
        T_start = 1.5 + 0.3 * (restart % 4)
        cooling = 0.99995 - 0.000005 * (restart % 6)
        min_T = 0.01
        
        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.85
        )
        
        # Greedy polish
        polished_colors, polished_cost = greedy_polish(sa_colors, sa_cost)
        
        # Pair swap polish for additional improvement
        if time.time() < deadline - 20:
            pair_colors, pair_cost = pair_swap_polish(polished_colors, polished_cost)
            if pair_cost < polished_cost:
                polished_colors, polished_cost = pair_colors, pair_cost
        
        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]
    
    # Final polish on the best result
    if global_best_colors is not None:
        final_colors, final_cost = greedy_polish(global_best_colors, global_best_cost)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
        
        # Additional pair swap polish if time allows
        if time.time() < deadline - 5:
            pair_colors, pair_cost = pair_swap_polish(global_best_colors, global_best_cost)
            if pair_cost < global_best_cost:
                global_best_cost = pair_cost
                global_best_colors = pair_colors
    
    return global_best_colors
