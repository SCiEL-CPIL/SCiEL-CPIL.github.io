# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    satisfying the kissing constraints, aiming to maximize k.

    Strategy:
      1. Restrict to equinorm vectors with squared norm R^2 = 4.
         This simplifies the condition min_dist >= max_norm to:
         For all pairs u, v: ||u-v||^2 >= 4  <=>  2*4 - 2*u.v >= 4  <=>  u.v <= 2.
      2. Generate all integer vectors of norm squared 4:
         - Type A: One coordinate is +/- 2, others 0.
         - Type B: Four coordinates are +/- 1, others 0.
      3. Use vectorized greedy construction with multiple random restarts.
      
    Returns:
        points: np.ndarray of shape (k, 11) with int64 entries.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    d = 11
    R2 = 4
    
    # Generate candidates
    candidates = []
    
    # Type A: +/- 2 at one position
    for i in range(d):
        for sign in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * sign
            candidates.append(v)
            
    # Type B: +/- 1 at four positions
    # Choose 4 indices out of 11
    indices = range(d)
    for pos_combo in itertools.combinations(indices, 4):
        # All 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for pos, sign in zip(pos_combo, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Convert to numpy array for vectorized operations
    # candidates is a list of arrays, let's keep it as list for easy appending in greedy, 
    # but we will check against a growing numpy array of selected points.
    
    # To speed up, we can pre-compute some properties or just rely on numpy broadcasting.
    
    best_points = None
    best_k = 0
    
    def greedy_build(candidates_list, max_time):
        selected = []
        selected_arr = np.empty((0, d), dtype=np.int64)
        
        for cand in candidates_list:
            if time.time() > max_time:
                break
            
            # Check if cand is valid against all selected points
            if len(selected) == 0:
                is_valid = True
            else:
                # Compute squared distances from cand to all selected points
                # selected_arr is (k, 11), cand is (11,)
                # diff = selected_arr - cand (broadcasting)
                # dist_sq = sum(diff**2, axis=1)
                # We need min(dist_sq) >= R2
                # Optimization: use dot products?
                # ||c - s||^2 = ||c||^2 + ||s||^2 - 2 c.s = 4 + 4 - 2 c.s = 8 - 2 c.s
                # Condition: 8 - 2 c.s >= 4  =>  4 >= 2 c.s  =>  c.s <= 2
                # So we just need to check if max dot product <= 2.
                dots = selected_arr @ cand
                is_valid = np.max(dots) <= 2
                
            if is_valid:
                selected.append(cand)
                # Append to array for next vectorized check
                selected_arr = np.vstack([selected_arr, cand])
                
        return selected

    n_restarts = 0
    rng = np.random.RandomState(42)
    
    while time.time() < deadline and n_restarts < 100:
        n_restarts += 1
        
        # Shuffle candidates
        shuffled_candidates = candidates.copy()
        rng.shuffle(shuffled_candidates)
        
        points = greedy_build(shuffled_candidates, deadline)
        
        if len(points) > best_k:
            best_k = len(points)
            best_points = points
            
    # Fallback if no points found (should not happen)
    if best_points is None or len(best_points) < 2:
        # Trivial valid set: (2,0,..) and (-2,0,..)
        v1 = np.zeros(d, dtype=np.int64); v1[0] = 2
        v2 = np.zeros(d, dtype=np.int64); v2[0] = -2
        best_points = [v1, v2]
        
    result = np.array(best_points, dtype=np.int64)
    result = np.around(result).astype(np.int64)
    return result

# EVOLVE-BLOCK-END
