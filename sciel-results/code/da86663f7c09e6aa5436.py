import time
import random
import numpy as np

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    # Precompute distance matrix
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords_flat = np.column_stack((X.ravel(), Y.ravel()))
    
    diff = coords_flat[:, np.newaxis, :] - coords_flat[np.newaxis, :, :]
    D = np.sum(diff * diff, axis=2).astype(np.uint16)
    
    best_set = []
    best_size = 0
    all_indices = list(range(TOTAL_POINTS))
    
    def is_valid_addition(k, s_indices, s_dist_sets):
        seen_dists_from_k = set()
        for i in s_indices:
            d = D[k, i]
            if d in seen_dists_from_k:
                return False
            seen_dists_from_k.add(d)
            if d in s_dist_sets[i]:
                return False
        return True

    def add_point(k, s_indices, s_dist_sets, in_set):
        new_dist_set = set()
        for i in s_indices:
            d = D[k, i]
            new_dist_set.add(d)
            s_dist_sets[i].add(d)
        s_indices.append(k)
        s_dist_sets.append(new_dist_set)
        in_set[k] = True

    def remove_point(k, s_indices, s_dist_sets, in_set):
        idx = s_indices.index(k)
        for i in s_indices:
            if i != k:
                d = D[k, i]
                s_dist_sets[s_indices.index(i)].discard(d)
        s_indices.pop(idx)
        s_dist_sets.pop(idx)
        in_set[k] = False

    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        shuffled = all_indices[:]
        random.shuffle(shuffled)
        
        s_indices = []
        s_dist_sets = []
        in_set = np.zeros(TOTAL_POINTS, dtype=bool)
        
        # Initial Greedy
        for k in shuffled:
            if time.time() > deadline:
                break
            if not in_set[k] and is_valid_addition(k, s_indices, s_dist_sets):
                add_point(k, s_indices, s_dist_sets, in_set)
        
        current_size = len(s_indices)
        
        # Local Search
        stuck_counter = 0
        max_stuck = 100
        local_search_iterations = 0
        max_local_iter = 2000
        
        while local_search_iterations < max_local_iter:
            if time.time() > deadline:
                break
            
            local_search_iterations += 1
            
            # Pick a random candidate
            k = random.choice(all_indices)
            
            if in_set[k]:
                continue
                
            if is_valid_addition(k, s_indices, s_dist_sets):
                add_point(k, s_indices, s_dist_sets, in_set)
                current_size = len(s_indices)
                stuck_counter = 0
            else:
                stuck_counter += 1
                
            if stuck_counter > max_stuck and s_indices:
                # Remove a random point to escape local minimum
                idx_to_remove = random.randint(0, len(s_indices) - 1)
                p_to_remove = s_indices[idx_to_remove]
                remove_point(p_to_remove, s_indices, s_dist_sets, in_set)
                current_size = len(s_indices)
                stuck_counter = 0
                
        if current_size > best_size:
            best_size = current_size
            # Convert to list of tuples of Python ints
            best_set = [tuple(coords_flat[i].tolist()) for i in s_indices]
            
    return best_set
