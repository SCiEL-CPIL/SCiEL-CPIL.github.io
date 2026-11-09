import random
import time
import numpy as np

N = 64
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
# dist_matrix[i, j] = squared distance between POINTS[i] and POINTS[j]
dist_matrix = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d2 = (xi - xj) ** 2 + (yi - yj) ** 2
        dist_matrix[i, j] = d2
        dist_matrix[j, i] = d2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = []
    best_size = 0
    
    indices = list(range(NUM_POINTS))
    
    # We will run multiple restarts
    # To diversify, we can use different shuffling strategies
    # 1. Pure random
    # 2. Random but biased towards corners/edges?
    
    max_restarts = 5000
    
    for seed in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        candidates = indices[:]
        random.shuffle(candidates)
        
        current_set = []
        # dist_sets[i] is a set of squared distances from current_set[i] to other points in current_set
        dist_sets = []
        
        for p_idx in candidates:
            valid = True
            d_to_p = []
            
            # Check against existing points
            for i, A_idx in enumerate(current_set):
                d2 = dist_matrix[p_idx, A_idx]
                d_to_p.append(d2)
                # Check if distance from A to P is already used by A to some other point
                if d2 in dist_sets[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if distances from P to all existing points are distinct
            # This prevents P from being the apex of an isosceles triangle
            if len(set(d_to_p)) != len(d_to_p):
                continue
            
            # If valid, add p
            # Update dist_sets for existing points
            for i, A_idx in enumerate(current_set):
                dist_sets[i].add(d_to_p[i])
            
            # Add new point
            current_set.append(p_idx)
            dist_sets.append(set(d_to_p))
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
    return [POINTS[i] for i in best_set]
# EVOLVE-BLOCK-END
