import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    norms_sq = np.sum(grid**2, axis=1)
    
    # Precompute distances to center for heuristics
    center = np.array([3.5, 3.5, 3.5])
    dist_to_center = np.sum((grid - center)**2, axis=1)
    
    def can_add_fast(arr, arr_norms, p, p_norm):
        m = len(arr)
        if m < 4:
            return True
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_norms - p_norm
        
        # Generate combinations of 4 indices
        idxs = np.array(list(itertools.combinations(range(m), 4)))
        if len(idxs) == 0:
            return True
            
        subs = w[idxs]
        dets = np.linalg.det(subs)
        
        return not np.any(np.abs(dets) < 0.5)

    def greedy_pass(order, time_limit):
        arr = np.empty((0, 3), dtype=np.int64)
        arr_norms = np.empty(0, dtype=np.int64)
        
        for p_idx in order:
            if time.time() > time_limit:
                break
                
            p = grid[p_idx]
            p_norm = norms_sq[p_idx]
            
            if can_add_fast(arr, arr_norms, p, p_norm):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_norms = np.array([p_norm], dtype=np.int64)
                else:
                    arr = np.vstack([arr, p])
                    arr_norms = np.append(arr_norms, p_norm)
                    
        return arr

    start_time = time.time()
    total_time_limit = 590.0
    
    best_set = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Precomputed orders for heuristics
    order_center_out = np.argsort(dist_to_center)
    order_corner_in = np.argsort(-dist_to_center) # Farthest from center first
    
    # We will run until time limit
    while time.time() < total_time_limit:
        # Choose a strategy
        r = rng.random()
        if r < 0.4:
            order = rng.permutation(N)
        elif r < 0.7:
            # Center out
            order = order_center_out
            # Add noise
            if rng.random() < 0.5:
                # Perturb
                order = order.copy()
                for i in range(N-1, 0, -1):
                    j = rng.integers(0, i+1)
                    order[i], order[j] = order[j], order[i]
        else:
            # Corner in
            order = order_corner_in
            if rng.random() < 0.5:
                order = order.copy()
                for i in range(N-1, 0, -1):
                    j = rng.integers(0, i+1)
                    order[i], order[j] = order[j], order[i]

        current_set = greedy_pass(order, total_time_limit)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    if best_set is None:
        return grid[:4].astype(np.int64)
        
    return best_set.astype(np.int64)
