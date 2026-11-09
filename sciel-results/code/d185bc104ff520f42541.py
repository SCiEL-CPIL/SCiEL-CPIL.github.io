import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    d = 11
    
    # Fallback: D_11 roots (220 points) - weight-2 vectors with all sign combos
    def get_d11_points():
        pts = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    pts.append(p)
        return np.array(pts)
    
    fallback_pts = get_d11_points()
    best_pts = fallback_pts
    best_k = fallback_pts.shape[0]
    
    # Try to build a larger set from weight-3 vectors
    # Vectors with exactly 3 non-zero entries, each +/- 1
    # Total: C(11, 3) * 2^3 = 165 * 8 = 1320
    
    all_vecs = []
    for indices in combinations(range(d), 3):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[indices[0]] = s1
                    p[indices[1]] = s2
                    p[indices[2]] = s3
                    all_vecs.append(p)
    
    all_vecs = np.array(all_vecs)
    
    # Greedy selection with distance threshold
    # For weight-3 vectors, norm_sq = 3.
    # We need min_dist_sq >= max_norm_sq.
    # If we mix weight-2 and weight-3, max_norm_sq = 3.
    # So we need all pairwise distances >= 3.
    # Two weight-3 vectors:
    # - If they share 0 indices: dist_sq = 6 (OK)
    # - If they share 1 index: dist_sq = 4 (OK)
    # - If they share 2 indices: dist_sq = 2 (NOT OK)
    # So we cannot have two weight-3 vectors that share 2 indices.
    
    # Let's try to greedily select weight-3 vectors with no two sharing 2 indices.
    # This is equivalent to a code with distance >= 4 (in terms of Hamming-like metric on support).
    
    selected = []
    selected_indices = []  # List of index tuples for fast conflict checking
    
    for p in all_vecs:
        if time.time() > deadline:
            break
            
        # Get the indices of non-zero entries
        idx = tuple(np.where(p != 0)[0])
        
        valid = True
        # Check against all selected points
        for sel_idx in selected_indices:
            # Count common indices
            common = len(set(idx) & set(sel_idx))
            if common >= 2:
                valid = False
                break
                
        if valid:
            selected.append(p)
            selected_indices.append(idx)
            
    if len(selected) > best_k:
        best_pts = np.array(selected)
        best_k = len(selected)
    
    # Now try to combine with some weight-2 vectors if possible
    # Actually, let's first verify the weight-3 set
    # Then try to add weight-2 vectors that are far enough from all weight-3 vectors
    
    # Check if we can do better by starting with weight-2 and adding weight-3
    # Or just stick with the greedy weight-3 set
    
    # Let's also try a different greedy: start with all weight-2, then add weight-3
    # But weight-2 vectors have norm_sq = 2, weight-3 have norm_sq = 3.
    # If we mix, max_norm_sq = 3, so all distances must be >= 3.
    # A weight-2 vector and a weight-3 vector:
    # - If weight-2 indices are subset of weight-3 indices (2 common): 
    #   dist_sq = (1-1)^2 + (1-1)^2 + (0-1)^2 + ... = 1 + 0 + 0 = 1? No.
    #   Let's compute: v2 = e_i + e_j, v3 = e_i + e_j + e_k
    #   diff = (0, 0, -1, ...) -> dist_sq = 1. This is < 3. NOT OK.
    #   If v2 = e_i - e_j, v3 = e_i + e_j + e_k:
    #   diff = (0, -2, -1, ...) -> dist_sq = 0 + 4 + 1 = 5. OK.
    #   If v2 = e_i + e_j, v3 = e_i - e_j + e_k:
    #   diff = (0, 2, -1, ...) -> dist_sq = 0 + 4 + 1 = 5. OK.
    #   If v2 = e_i - e_j, v3 = e_i - e_j + e_k:
    #   diff = (0, 0, -1, ...) -> dist_sq = 1. NOT OK.
    
    # So mixing is complex. Let's just try to maximize within weight-3.
    # The greedy above gives a code with no two sharing 2 indices.
    # This is a constant weight code with d=11, w=3, distance >= 4 (in Hamming).
    # The maximum size of such a code is known as A(11, 4, 3) or similar.
    
    # Let's verify the greedy result
    if best_pts.shape[1] != d:
        return fallback_pts
        
    k = best_pts.shape[0]
    if k < 2:
        return fallback_pts
    
    # Check no zero vector
    if np.any(np.all(best_pts == 0, axis=1)):
        mask = np.any(best_pts != 0, axis=1)
        best_pts = best_pts[mask]
        k = best_pts.shape[0]
        
    # Check min dist sq >= max norm sq
    max_norm_sq = np.max(np.sum(best_pts**2, axis=1))
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i+1, k):
            diff = best_pts[i] - best_pts[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq < max_norm_sq:
                # Invalid, return fallback
                return fallback_pts
                
    return best_pts
