import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer lattice points using a greedy
    farthest-point sampling strategy over a pool of low-norm integer vectors.
    This attempts to find a configuration larger than the static weight-2 set (220 points)
    by potentially mixing norms or finding a denser packing on a specific norm shell.
    """
    deadline = time.time() + 590.0  # Safety margin
    
    d = 11
    
    # Generate candidate points
    # We focus on norms squared: 1, 2, 4, 5, 8, 9, 10
    # To keep the pool manageable, we restrict coordinates to small integers.
    
    candidates = []
    
    # Norm 1: +/- e_i
    for i in range(d):
        for s in (-1, 1):
            p = np.zeros(d, dtype=np.int64)
            p[i] = s
            candidates.append(p)
            
    # Norm 2: +/- e_i +/- e_j
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    candidates.append(p)
                    
    # Norm 4: 
    # Type 1: One 2, rest 0. (2^2 = 4)
    for i in range(d):
        for s in (-1, 1):
            p = np.zeros(d, dtype=np.int64)
            p[i] = 2 * s
            candidates.append(p)
    # Type 2: Four 1s. (1+1+1+1=4)
    # This generates many. C(11,4) * 16 = 330 * 16 = 5280.
    # Let's include a subset or all if time permits.
    # For now, let's stick to Norm 1 and 2 to ensure speed, 
    # but the greedy approach might allow swapping.
    
    # Actually, let's just use the Norm 2 set as the base, 
    # but try to ADD Norm 1 points if they fit, or REMOVE conflicting Norm 2 points.
    
    # Better strategy: Greedy from scratch on the combined pool of Norm 1 and Norm 2.
    # Pool size: 22 + 220 = 242.
    # This is small enough for O(N^2) or even O(N^3) in Python if optimized, 
    # but we can use NumPy.
    
    cand_array = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_array)
    
    # Precompute norms squared
    cand_norms = np.sum(cand_array * cand_array, axis=1)
    
    # Greedy selection
    selected_indices = []
    selected_points = []
    
    # Start with a random point to break symmetry
    start_idx = random.randint(0, n_cands - 1)
    selected_indices.append(start_idx)
    selected_points.append(cand_array[start_idx])
    
    current_max_norm = cand_norms[start_idx]
    current_min_dist = float('inf')
    
    while time.time() < deadline:
        if not selected_indices:
            break
            
        S = np.array(selected_points, dtype=np.int64)
        k = len(S)
        
        # Compute distances from all candidates to all selected points
        # D[i, j] = ||cand[i] - S[j]||^2
        # D = ||cand||^2 + ||S||^2 - 2 * cand @ S.T
        
        cand_norms_col = cand_norms[:, np.newaxis]
        S_norms_row = np.sum(S * S, axis=1)[np.newaxis, :]
        
        # Dot products
        dots = cand_array @ S.T
        
        dists = cand_norms_col + S_norms_row - 2 * dots
        
        # For each candidate, find the min distance to any selected point
        min_dists = np.min(dists, axis=1)
        
        # The constraint is: min_dist >= max_norm
        # max_norm is max(current_max_norm, norm of new candidate)
        
        # We want to pick the candidate that maximizes the "margin" or just the min_dist,
        # subject to validity.
        
        # Find candidates that are valid:
        # valid if min_dist >= max(current_max_norm, cand_norms[i])
        
        valid_mask = np.zeros(n_cands, dtype=bool)
        for i in range(n_cands):
            if i in selected_indices:
                continue
            req_norm = max(current_max_norm, cand_norms[i])
            if min_dists[i] >= req_norm:
                valid_mask[i] = True
        
        if not np.any(valid_mask):
            break
        
        # Among valid candidates, pick the one with the largest min_dist
        # This is a greedy farthest-point heuristic
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            break
            
        # Get min_dists for valid candidates
        valid_min_dists = min_dists[valid_indices]
        
        # Pick the index with the max min_dist
        best_local_idx = np.argmax(valid_min_dists)
        best_global_idx = valid_indices[best_local_idx]
        
        # Add to selected
        selected_indices.append(best_global_idx)
        selected_points.append(cand_array[best_global_idx])
        
        # Update max norm
        current_max_norm = max(current_max_norm, cand_norms[best_global_idx])
        
        # Optional: If we want to be faster, we can stop early if we exceed 220
        # but we want the max possible.
        
    # Final verification and formatting
    if len(selected_points) < 2:
        # Fallback to the known good 220-point set if greedy failed
        # Regenerate the standard weight-2 set
        points_list = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = si
                        p[j] = sj
                        points_list.append(p)
        points = np.array(points_list, dtype=np.int64)
    else:
        points = np.array(selected_points, dtype=np.int64)
        
    # Round and cast
    points = np.around(points).astype(np.int64)
    
    # Final safety check
    k = len(points)
    if k >= 2:
        # Verify constraints one last time
        max_norm_sq = 0
        for p in points:
            ns = int(np.dot(p, p))
            if ns > max_norm_sq:
                max_norm_sq = ns
        
        min_dist_sq = float('inf')
        for i in range(k):
            for j in range(i + 1, k):
                diff = points[i] - points[j]
                ds = int(np.dot(diff, diff))
                if ds < min_dist_sq:
                    min_dist_sq = ds
        
        if min_dist_sq < max_norm_sq:
            # Invalid, fallback
            points_list = []
            for i in range(d):
                for j in range(i + 1, d):
                    for si in (-1, 1):
                        for sj in (-1, 1):
                            p = np.zeros(d, dtype=np.int64)
                            p[i] = si
                            p[j] = sj
                            points_list.append(p)
            points = np.array(points_list, dtype=np.int64)
            points = np.around(points).astype(np.int64)

    return points
