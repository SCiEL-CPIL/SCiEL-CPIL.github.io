import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    # 1. Precomputation
    # Map (i,j) to index
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edge_list.append((i, j))
            idx += 1

    # Generate all 4-cycles
    # A 4-cycle on vertices a,b,c,d is a sequence of 4 edges.
    # For a set {a,b,c,d}, the 3 distinct 4-cycles are:
    # 1: a-b-c-d-a
    # 2: a-b-d-c-a
    # 3: a-c-b-d-a
    all_4sets = []
    cycles = [] # List of tuples of 4 edge indices
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    all_4sets.append((a, b, c, d))
                    # Cycle 1: a-b, b-c, c-d, d-a
                    cycles.append((
                        edge_index[(a, b)], 
                        edge_index[(b, c)], 
                        edge_index[(c, d)], 
                        edge_index[(d, a)]
                    ))
                    # Cycle 2: a-b, b-d, d-c, c-a
                    cycles.append((
                        edge_index[(a, b)], 
                        edge_index[(b, d)], 
                        edge_index[(d, c)], 
                        edge_index[(c, a)]
                    ))
                    # Cycle 3: a-c, c-b, b-d, d-a
                    cycles.append((
                        edge_index[(a, c)], 
                        edge_index[(c, b)], 
                        edge_index[(b, d)], 
                        edge_index[(d, a)]
                    ))
    
    num_cycles = len(cycles)
    
    # Map each edge to the cycles it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for c_idx, cycle in enumerate(cycles):
        for e_idx in cycle:
            edge_to_cycles[e_idx].append(c_idx)

    # 2. Initialize Coloring
    # Base: c(i,j) = min(i,j)
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    for idx, (i, j) in enumerate(edge_list):
        colors[idx] = i
        
    # Initial violation count
    def check_cycle_valid(cycle, cols):
        c0, c1, c2, c3 = cols[cycle[0]], cols[cycle[1]], cols[cycle[2]], cols[cycle[3]]
        # Rainbow if all distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return 1
        return 0

    # Precompute cycle colors as numpy arrays for faster access? 
    # cycles is a list of tuples. We can convert to a 2D numpy array for vectorized checks if needed, 
    # but for incremental updates, we just need to check specific cycles.
    
    cycle_colors = np.array(cycles, dtype=np.int32) # Shape (num_cycles, 4)
    
    def get_violations(cols):
        # cols: numpy array of shape (NUM_EDGES,)
        # cycle_colors: shape (num_cycles, 4)
        # Get colors for all edges in all cycles: shape (num_cycles, 4)
        cc = cols[cycle_colors]
        # Check if all 4 are distinct
        # c0!=c1, c0!=c2, c0!=c3, c1!=c2, c1!=c3, c2!=c3
        c0, c1, c2, c3 = cc[:,0], cc[:,1], cc[:,2], cc[:,3]
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        return distinct.sum()

    current_violations = get_violations(colors)
    
    # Track distinct colors
    color_counts = np.zeros(MAX_COLOR, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
    current_distinct = np.count_nonzero(color_counts)
    
    best_colors = colors.copy()
    best_distinct = current_distinct
    
    # 3. Simulated Annealing
    T_start = 10.0
    T_end = 0.01
    num_iterations = 2000000 # High number, will break on time
    
    # Helper to check if a specific cycle is a rainbow
    def is_rainbow(c_idx, cols):
        c0, c1, c2, c3 = cols[cycle_colors[c_idx, 0]], cols[cycle_colors[c_idx, 1]], cols[cycle_colors[c_idx, 2]], cols[cycle_colors[c_idx, 3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return 1
        return 0

    for it in range(num_iterations):
        if time.time() - start_time > DEADLINE:
            break
            
        # Temperature
        progress = it / num_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e_idx]
        
        # Choose new color
        # Strategy: 50% random from 0..119, 50% from existing colors
        # Actually, to maximize distinct colors, we should bias towards unused colors or low-index unused colors?
        # No, random is better for exploration. But if we pick a color > current max, it increases distinct count.
        # Let's just pick random 0..119.
        new_color = random.randint(0, MAX_COLOR - 1)
        
        if new_color == old_color:
            continue
            
        # Calculate delta violations
        # Only cycles containing e_idx are affected
        delta_viol = 0
        for c_idx in edge_to_cycles[e_idx]:
            # Check old state
            old_rainbow = is_rainbow(c_idx, colors)
            # Temporarily change
            colors[e_idx] = new_color
            new_rainbow = is_rainbow(c_idx, colors)
            # Restore
            colors[e_idx] = old_color
            
            if old_rainbow == 0 and new_rainbow == 1:
                delta_viol += 1
            elif old_rainbow == 1 and new_rainbow == 0:
                delta_viol -= 1
        
        # Calculate delta distinct
        delta_distinct = 0
        if color_counts[old_color] == 1:
            delta_distinct -= 1
        if color_counts[new_color] == 0:
            delta_distinct += 1
            
        # Objective: minimize violations, maximize distinct
        # We prioritize validity. If violations > 0, we want to reduce them.
        # If violations == 0, we want to increase distinct.
        # Energy function: E = violations * 100 - distinct
        delta_E = delta_viol * 100 - delta_distinct
        
        # Acceptance
        if delta_E < 0:
            accept = True
        else:
            # Metropolis criterion
            # exp(-delta_E / T)
            # If T is small and delta_E is large, prob is 0
            if T > 0:
                accept = random.random() < np.exp(-delta_E / T)
            else:
                accept = False
                
        if accept:
            colors[e_idx] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            current_violations += delta_viol
            current_distinct += delta_distinct
            
            if current_violations == 0 and current_distinct > best_distinct:
                best_distinct = current_distinct
                best_colors = colors.copy()
                
    # Final check and return
    # Ensure best_colors is valid
    final_violations = get_violations(best_colors)
    if final_violations != 0:
        # Fallback to a known valid construction if SA failed to find a valid one
        # This is the base coloring from the current program
        fallback = np.zeros(NUM_EDGES, dtype=np.int32)
        for idx, (i, j) in enumerate(edge_list):
            fallback[idx] = i
        fallback[edge_idx(13, 15)] = 15
        fallback[edge_idx(14, 15)] = 16
        return fallback.tolist()
        
    return best_colors.tolist()
