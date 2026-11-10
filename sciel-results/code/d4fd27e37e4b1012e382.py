# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points satisfying:
      - shape (k, 11), k >= 2
      - no zero vector
      - min pairwise squared distance >= max squared norm
    Maximizes k.
    """
    start_time = time.time()
    deadline = 630.0  # seconds
    d = 11

    def elapsed_ok():
        return (time.time() - start_time) < (deadline - 5.0)

    # Strategy: use the standard construction from coding theory / sphere packing.
    # In dimension 11, we can use vectors from the root lattice E11 or simpler constructions.
    # 
    # A well-known construction: take all vectors of the form (±1, ±1, ..., ±1) in R^11.
    # There are 2^11 = 2048 such vectors. Each has norm^2 = 11.
    # The squared distance between two such vectors: if they differ in t coordinates,
    # distance^2 = 4t. Minimum t=2 (since they're distinct and not negatives? wait).
    # Actually two distinct sign vectors differ in at least 1 coordinate.
    # If they differ in 1 coordinate: dist^2 = 4. But 4 < 11, so that fails.
    # So we need a subset where any two differ in at least ceil(11/4) = 3 coordinates? 
    # We need 4t >= 11 => t >= 3 (since 4*2=8 < 11, 4*3=12 >= 11).
    # So we need a binary code of length 11 with minimum Hamming distance 3.
    # The maximum size of such a code is A(11, 3). 
    # A(11,3) is known to be 1024 (using a shortened Hamming code or similar).
    # Actually, for binary codes, A(n,3) = 2^(n - ceil(log2(n+1))) for Hamming bound tight cases.
    # For n=11: ceil(log2(12)) = 4, so 2^(11-4) = 128. That's the Hamming bound.
    # But actually A(11,3) is larger. Let me think differently.
    #
    # Alternative: use vectors with entries in {0, ±1} or other constructions.
    #
    # Let's try a different approach: use the construction from the paper on kissing numbers.
    # In dimension 11, the kissing number is known to be at least 532.
    # A construction achieving this uses the E11 lattice or a related construction.
    #
    # Simpler approach: use all vectors in {−1, 0, 1}^11 with exactly 3 nonzero entries,
    # and ensure pairwise distances are large enough.
    #
    # Let me try a greedy approach with a good initial set and then expand.

    # Start with a known valid set: use vectors with exactly one +1 and one -1, rest 0.
    # Number of such vectors: C(11,2) * 2 = 55 * 2 = 110.
    # Each has norm^2 = 2.
    # Distance between two such vectors: depends on overlap.
    # If they share no indices: dist^2 = 4 (each contributes 1^2+1^2=2, no overlap)
    # Wait: vector a has +1 at i, -1 at j. Vector b has +1 at k, -1 at l.
    # If {i,j} ∩ {k,l} = ∅: dist^2 = 4 (4 nonzero entries each with diff 1 or -1... let me compute)
    # a_i = 1, a_j = -1, a_k = 0, a_l = 0
    # b_i = 0, b_j = 0, b_k = 1, b_l = -1
    # (a-b)_i = 1, (a-b)_j = -1, (a-b)_k = -1, (a-b)_l = 1 => dist^2 = 4
    # 4 >= 2 ✓
    # If they share one index, say i=k: a_i=1, b_i=1 => diff 0. 
    # a_j=-1, b_j=0 => diff -1. a_l=0, b_l=-1 => diff 1. 
    # So dist^2 = 0 + 1 + 1 + ... = 2. 2 >= 2 ✓ (equal is fine)
    # If they share both indices but swapped: a_i=1, a_j=-1; b_i=-1, b_j=1.
    # diff_i = 2, diff_j = -2 => dist^2 = 8. ✓
    # So all 110 vectors work! Max norm^2 = 2, min dist^2 = 2.
    
    # Can we add more? Try adding vectors with 3 nonzero entries.
    # Vector with three ±1's: norm^2 = 3. But max norm becomes 3, so we need min dist^2 >= 3.
    # Distance between a 2-nonzero vector and a 3-nonzero vector could be small.
    # This gets complicated. Let's stick with the 110-vector construction and see if we can improve.
    
    # Generate all ±e_i ± e_j vectors
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = -1
            points.append(v1)
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = -1
            v2[j] = 1
            points.append(v2)
    
    points_arr = np.array(points, dtype=np.int64)
    
    # Verify the construction
    def is_valid(pts):
        if len(pts) < 2:
            return False
        if pts.shape[1] != 11:
            return False
        if np.any(pts == 0):  # This checks if any element is 0, but we need to check if any ROW is zero
            pass
        # Check no zero vector
        for p in pts:
            if np.all(p == 0):
                return False
        # Compute max norm squared
        max_norm_sq = 0
        for p in pts:
            norm_sq = np.dot(p, p)
            if norm_sq > max_norm_sq:
                max_norm_sq = norm_sq
        # Compute min pairwise distance squared
        min_dist_sq = float('inf')
        k = len(pts)
        for i in range(k):
            for j in range(i + 1, k):
                diff = pts[i] - pts[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        return min_dist_sq >= max_norm_sq
    
    if not is_valid(points_arr):
        # Fallback to the trivial 2-point solution
        return np.array([np.ones(11), -np.ones(11)], dtype=np.int64)
    
    # Try to add more points using a greedy approach
    # Candidate points: all vectors in {-1, 0, 1}^11 with small weight
    # We'll try to add vectors that maintain the validity condition
    
    # Current max norm squared
    max_norm_sq = int(np.max(np.sum(points_arr ** 2, axis=1)))
    
    # Generate candidate points: all vectors with entries in {-1, 0, 1} and weight <= 4
    # But that's 3^11 = 177147 candidates, which is manageable
    # Let's be more selective: try weight 3 and 4 vectors
    
    candidates = []
    # Weight 3 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for signs in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                              (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = signs[0]
                    v[j] = signs[1]
                    v[k] = signs[2]
                    candidates.append(v)
    
    # Also try weight 1 vectors (just ±e_i)
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Remove candidates that are already in points
    current_set = set([tuple(p) for p in points_arr])
    new_candidates = []
    for c in candidates:
        if tuple(c) not in current_set:
            new_candidates.append(c)
    
    # Greedily add candidates
    added = 0
    for c in new_candidates:
        if not elapsed_ok():
            break
        test_pts = np.vstack([points_arr, c.reshape(1, -1)])
        if is_valid(test_pts):
            points_arr = test_pts
            added += 1
            # Update max_norm_sq
            c_norm_sq = int(np.dot(c, c))
            if c_norm_sq > max_norm_sq:
                max_norm_sq = c_norm_sq
    
    # Try another round with weight 4 vectors if time permits
    if elapsed_ok() and added > 0:
        candidates4 = []
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for l in range(k + 1, d):
                        # Only try all +1 and one -1 to keep it manageable
                        for signs in [(1, 1, 1, 1), (1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1)]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = signs[0]
                            v[j] = signs[1]
                            v[k] = signs[2]
                            v[l] = signs[3]
                            candidates4.append(v)
        
        current_set = set([tuple(p) for p in points_arr])
        for c in candidates4:
            if not elapsed_ok():
                break
            if tuple(c) not in current_set:
                test_pts = np.vstack([points_arr, c.reshape(1, -1)])
                if is_valid(test_pts):
                    points_arr = test_pts
                    current_set.add(tuple(c))
    
    # Final validation
    if not is_valid(points_arr):
        return np.array([np.ones(11), -np.ones(11)], dtype=np.int64)
    
    return points_arr


# EVOLVE-BLOCK-END
