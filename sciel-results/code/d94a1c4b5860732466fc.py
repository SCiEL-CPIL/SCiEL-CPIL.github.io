import time
import random
import math
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute cycles
CYCLES = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                # Cycle 1: a-b-c-d-a
                CYCLES.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                # Cycle 2: a-b-d-c-a
                CYCLES.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                # Cycle 3: a-c-b-d-a
                CYCLES.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

NUM_CYCLES = len(CYCLES)
cycle_edges = np.array(CYCLES, dtype=np.int32)

# Precompute edge to cycle mapping
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for c_idx in range(NUM_CYCLES):
    for e in cycle_edges[c_idx]:
        edge_to_cycles[e].append(c_idx)

# Convert to numpy arrays for faster access
edge_to_cycle_indices = [np.array(lst, dtype=np.int32) for lst in edge_to_cycles]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_score = 0
    best_colors = None
    
    # Multi-start
    while time.time() < deadline:
        # Initialize
        current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                current_colors[idx] = i
                idx += 1
        
        global_color_counts = np.zeros(MAX_COLOR, dtype=np.int32)
        for c in current_colors:
            global_color_counts[c] += 1
            
        current_score = np.count_nonzero(global_color_counts)
        
        if current_score > best_score:
            best_score = current_score
            best_colors = current_colors.copy()
            
        T = 5.0
        T_min = 0.01
        alpha = 0.9995
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            e = random.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Random color
            new_color = random.randint(0, MAX_COLOR - 1)
            
            if new_color == old_color:
                continue
                
            # Check validity using vectorized operations
            affected_indices = edge_to_cycle_indices[e]
            if len(affected_indices) > 0:
                # Get edges in these cycles
                edges_in_cycles = cycle_edges[affected_indices]
                
                # Temporarily update
                current_colors[e] = new_color
                
                # Get current colors for these edges
                colors_in_cycles = current_colors[edges_in_cycles]
                
                # Check for rainbow (4 distinct colors)
                sorted_colors = np.sort(colors_in_cycles, axis=1)
                is_rainbow = (sorted_colors[:, 0] != sorted_colors[:, 1]) & \
                             (sorted_colors[:, 1] != sorted_colors[:, 2]) & \
                             (sorted_colors[:, 2] != sorted_colors[:, 3])
                
                valid = not np.any(is_rainbow)
            else:
                valid = True
                
            if valid:
                # Update global counts and score
                old_count = global_color_counts[old_color]
                new_count = global_color_counts[new_color]
                
                delta_score = 0
                if new_count == 0: delta_score += 1
                if old_count == 1: delta_score -= 1
                
                new_score = current_score + delta_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_colors = current_colors.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Acceptance criterion
                if delta_score >= 0:
                    current_score = new_score
                    global_color_counts[old_color] -= 1
                    global_color_counts[new_color] += 1
                else:
                    if random.random() < math.exp(delta_score / T):
                        current_score = new_score
                        global_color_counts[old_color] -= 1
                        global_color_counts[new_color] += 1
                    else:
                        # Revert
                        current_colors[e] = old_color
            else:
                # Revert
                current_colors[e] = old_color
                no_improve_count += 1
                
            T *= alpha
            if T < T_min:
                T = T_min
                
        # If we exited due to no_improve, the outer while loop will restart with a new seed (implicitly by re-initializing)
        # Note: The current initialization is deterministic (color(i,j)=i). 
        # To make multi-start effective, we should perturb the initial state or use random initialization for subsequent runs.
        # However, random initialization is likely invalid.
        # So we stick to the deterministic start, but the SA path is random due to random moves.
        # To add diversity, we can slightly perturb the initial state in subsequent runs if time permits.
        # For now, the random moves provide diversity.
        
    return best_colors.tolist() if best_colors is not None else [0]*NUM_EDGES
