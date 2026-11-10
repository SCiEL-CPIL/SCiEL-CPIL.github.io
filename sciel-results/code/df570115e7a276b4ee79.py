# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than or equal to their minimum pairwise distance, aiming to maximize the number of points.

    Strategy:
    1. Start with all weight-2 vectors (220 points, norm^2=2).
    2. Greedily add weight-4 vectors (norm^2=4) if they satisfy:
       - For all existing points p: ||p - v||^2 >= max_norm^2
       - For all existing weight-4 points w: ||w - v||^2 >= max_norm^2
       
    The key constraint is that for any two points u, v in our set:
       ||u - v||^2 >= max(||u||^2, ||v||^2)
    
    For weight-2 vectors (norm^2=2) and weight-4 vectors (norm^2=4):
       ||u - v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 2 + 4 - 2*u.v = 6 - 2*u.v
       We need 6 - 2*u.v >= 4, so u.v <= 1.
    
    For two weight-4 vectors u, v:
       ||u - v||^2 = 4 + 4 - 2*u.v = 8 - 2*u.v
       We need 8 - 2*u.v >= 4, so u.v <= 2.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    d = 11
    
    # Step 1: Generate all weight-2 vectors
    w2_vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    w2_vectors.append(v)
    
    # Step 2: Generate all weight-4 vectors
    w4_vectors = []
    for indices in combinations(range(d), 4):
        for signs in product([1, -1], repeat=4):
            v = [0] * d
            for idx, s in zip(indices, signs):
                v[idx] = s
            w4_vectors.append(v)
    
    if time.time() > deadline - 5:
        return np.array(w2_vectors, dtype=np.int64)
    
    # Start with all weight-2 vectors
    selected = list(w2_vectors)
    selected_norms2 = [2] * len(selected)
    max_norm2 = 2
    
    # Greedily add weight-4 vectors
    # For a weight-4 vector v to be added:
    #   - For all w2 vectors u: u.v <= 1
    #   - For all selected w4 vectors w: w.v <= 2
    
    # Precompute dot products of w4 vectors with all w2 vectors
    w2_arr = np.array(w2_vectors, dtype=np.int64)
    w4_arr = np.array(w4_vectors, dtype=np.int64)
    
    # Dot products of each w4 with all w2: shape (len(w4), len(w2))
    w4_w2_dots = w4_arr @ w2_arr.T
    
    # For each w4, check if all w2 dots are <= 1
    w4_valid_w2 = np.all(w4_w2_dots <= 1, axis=1)
    
    # Filter w4 vectors that pass the w2 constraint
    candidate_w4 = [w4_vectors[i] for i in range(len(w4_vectors)) if w4_valid_w2[i]]
    
    if time.time() > deadline - 5:
        return np.array(selected, dtype=np.int64)
    
    # Greedily add w4 vectors that satisfy pairwise constraint with already selected w4
    selected_w4 = []
    selected_w4_arr = []
    
    for v in candidate_w4:
        if time.time() > deadline - 10:
            break
        
        v_arr = np.array(v, dtype=np.int64)
        
        # Check against all selected w4 vectors
        valid = True
        if selected_w4_arr:
            dots = selected_w4_arr @ v_arr
            if np.any(dots > 2):
                valid = False
        
        if valid:
            selected_w4.append(v)
            if not selected_w4_arr:
                selected_w4_arr = v_arr.reshape(1, -1)
            else:
                selected_w4_arr = np.vstack([selected_w4_arr, v_arr.reshape(1, -1)])
    
    # Combine
    all_points = selected + selected_w4
    
    if len(all_points) < 2:
        return np.array(selected, dtype=np.int64)
    
    points = np.array(all_points, dtype=np.int64)
    
    # Final verification
    k = points.shape[0]
    norms2 = np.sum(points**2, axis=1)
    max_norm2 = np.max(norms2)
    
    min_dist2 = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d2 = int(np.dot(diff, diff))
            if d2 < min_dist2:
                min_dist2 = d2
        if time.time() > deadline - 5:
            break
    
    if np.any(np.all(points == 0, axis=1)):
        return np.array(selected, dtype=np.int64)
    
    if min_dist2 < max_norm2:
        # Fallback to just w2 vectors
        points = np.array(selected, dtype=np.int64)
    
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
