"""
Strategy: 
1. Population-Based Multi-Start Simulated Annealing.
2. Hard Constraint Satisfaction: Only valid colorings are considered.
3. Objective: Maximize the number of distinct colors.
4. Escape Mechanism: Accept moves that reduce distinct colors with a probability 
   dependent on temperature to escape local minima.
5. Efficient Validity Check: Vectorized NumPy check.
"""

import time
import numpy as np
import random
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge indices and cycle structures
def _get_edge_indices():
    """Return edge index mapping and cycle structures."""
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1
    
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((idx_map[(a,b)], idx_map[(b,c)], idx_map[(c,d)], idx_map[(d,a)]))
        # Cycle 2: a-b-d-c-a
        cycles.append((idx_map[(a,b)], idx_map[(b,d)], idx_map[(c,d)], idx_map[(a,c)]))
        # Cycle 3: a-c-b-d-a
        cycles.append((idx_map[(a,c)], idx_map[(b,c)], idx_map[(b,d)], idx_map[(a,d)]))
    
    cycles_np = np.array(cycles, dtype=np.int32)
    
    return idx_map, cycles_np

IDX_MAP, CYCLES = _get_edge_indices()
NUM_CYCLES = len(CYCLES)

def _is_valid_vectorized(colors_np):
    """
    Check if colors_np (shape NEDGES,) is valid.
    Returns True if no cycle has 4 distinct colors.
    """
    cycle_colors = colors_np[CYCLES]
    # A cycle is invalid if all 4 colors are distinct.
    # Sort the 4 colors. If they are all distinct, sorted[0] < sorted[1] < sorted[2] < sorted[3].
    # If any two are equal, it's valid.
    sorted_colors = np.sort(cycle_colors, axis=1)
    # Check if any adjacent pair in sorted order is equal
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return bool(np.all(adjacent_equal))

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def _initial_valid_coloring(rng):
    """
    Generate a valid 2-coloring.
    Split vertices into two sets of 8.
    Edges within set 0: color 0
    Edges within set 1: color 1
    Edges between sets: color 0
    This is valid because any 4-cycle will have at most 2 colors.
    """
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
    return colors

def _simulated_annealing(colors, deadline, rng):
    """
    Perform Simulated Annealing on a single coloring.
    """
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    
    current_colors = colors.copy()
    current_distinct = best_distinct
    
    # Maintain color counts for fast distinct count update
    color_counts = np.bincount(current_colors, minlength=MAX_COLOR)
    distinct_count = np.sum(color_counts > 0)
    unused_set = set(c for c in range(MAX_COLOR) if color_counts[c] == 0)
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Pick a random edge
        e = rng.randint(0, NEDGES - 1)
        
        old_color = current_colors[e]
        
        # Select new color
        if len(unused_set) > 0 and rng.random() < 0.8:
            new_color = rng.choice(list(unused_set))
        else:
            new_color = rng.randint(0, MAX_COLOR - 1)
            
        if new_color == old_color:
            continue
            
        # Apply move
        current_colors[e] = new_color
        
        # Check validity
        if _is_valid_vectorized(current_colors):
            # Update state
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            if color_counts[old_color] == 0:
                unused_set.add(old_color)
                distinct_count -= 1
            if color_counts[new_color] == 1:
                unused_set.discard(new_color)
                distinct_count += 1
                
            delta = distinct_count - current_distinct
            current_distinct = distinct_count
            
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_colors = current_colors.copy()
                
            if delta < 0:
                # Check acceptance
                prob = np.exp(delta / T)
                if rng.random() >= prob:
                    # Revert
                    current_colors[e] = old_color
                    color_counts[old_color] += 1
                    color_counts[new_color] -= 1
                    
                    if color_counts[old_color] == 1:
                        unused_set.discard(old_color)
                        distinct_count += 1
                    if color_counts[new_color] == 0:
                        unused_set.add(new_color)
                        distinct_count -= 1
                    current_distinct = distinct_count
        else:
            # Invalid, revert
            current_colors[e] = old_color
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    rng = random.Random(42)
    
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Generate initial valid coloring
        colors = _initial_valid_coloring(rng)
        
        # Add some random valid perturbations to diversify the starting point
        for _ in range(rng.randint(10, 50)):
            e = rng.randint(0, NEDGES - 1)
            new_color = rng.randint(2, MAX_COLOR - 1)
            colors[e] = new_color
            if _is_valid_vectorized(colors):
                pass
            else:
                colors[e] = 0 # Revert to safe
        
        # Determine time for this SA run
        local_deadline = min(time.time() + 10.0, deadline - 2.0)
        
        colors, distinct = _simulated_annealing(colors, local_deadline, rng)
        
        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()
            
        num_restarts += 1
        
        if num_restarts >= 5 and time.time() > deadline - 10.0:
            break

    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring(rng)
    
    return best_overall_colors.tolist()

# EVOLVE-BLOCK-END
