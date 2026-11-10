# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large set of points in the 8x8x8 grid such that no 5 points are co-spherical.
    Uses a combination of randomized greedy construction and local search (swaps)
    with a strict deadline guard.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms for efficiency
    sq_norms = (grid ** 2).sum(axis=1)
    
    def is_valid_set(points):
        """Check if a set of points is valid (no 5 co-spherical)."""
        # This is used for final verification, but we assume construction maintains validity.
        # We can skip this for speed if we trust the incremental check.
        pass

    def can_add(points, p_idx):
        """
        Check if adding point p_idx to the set 'points' (array of indices) is valid.
        points: array of indices into 'grid'
        p_idx: index of candidate point
        """
        m = len(points)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = sq_norms[p_idx]
        
        # We need to check all combinations of 4 points from 'points'
        # to see if they form a co-spherical set with p.
        # The condition is that the 4x4 determinant of 
        # rows (q - p, |q|^2 - |p|^2) is 0.
        
        # To speed this up, we can precompute the relative vectors for the current set?
        # No, the set changes.
        # But we can precompute the lifted coordinates.
        
        # Let's use the relative determinant check.
        # For each combo of 4 points from 'points':
        #   q1, q2, q3, q4
        #   vec_i = q_i - p
        #   diff_sq_i = |q_i|^2 - |p|^2
        #   det([vec_1, diff_sq_1], ...) != 0
        
        # Optimization: If m is large, this is slow.
        # But m is expected to be < 50. C(50,4) ~ 200k.
        # 200k determinants of 4x4 is ~ 200k * 16 ops ~ 3M ops.
        # This is fast enough in numpy if vectorized?
        # np.linalg.det is not vectorized over the last two dims efficiently for many small matrices?
        # Actually, np.linalg.det works on (..., M, M).
        
        if m == 4:
            q_indices = points
            qs = grid[q_indices]
            q_sqs = sq_norms[q_indices]
            
            # Build 4x4 matrix
            # Rows: q_i - p, |q_i|^2 - |p|^2
            mat = np.empty((4, 4), dtype=np.float64)
            mat[:, 0:3] = qs - p
            mat[:, 3] = q_sqs - p_sq
            
            return abs(np.linalg.det(mat)) >= 0.5

        # For m > 4, we need to check all C(m, 4) combos.
        # This is the bottleneck.
        # We can iterate through combinations.
        
        # To make it faster, we can early exit if we find a violation.
        
        # Generate combinations of indices from 'points'
        # itertools.combinations is slow in Python loop.
        # We can generate the matrix of all combinations?
        # C(m, 4) can be large.
        
        # Alternative: 
        # If we maintain the set as a list, and we are adding points one by one,
        # we only need to check the new point against the existing set.
        
        # Let's try to vectorize the check for a given p.
        # We have m points. We need to check all 4-subsets.
        # This is hard to vectorize without generating all subsets.
        
        # Given the small N (512) and expected k (< 50), 
        # a Python loop over combinations might be acceptable if we are careful,
        # or we can use a simpler heuristic:
        # Check against the last 4 points? No, must be all.
        
        # Let's stick to the combination loop but optimize the inner check.
        
        # Pre-compute the relative vectors for all points in 'points' relative to p?
        # rel_vecs[i] = grid[points[i]] - p
        # rel_sqs[i] = sq_norms[points[i]] - p_sq
        
        rel_vecs = grid[points] - p
        rel_sqs = sq_norms[points] - p_sq
        
        # Now check all combos of 4 from range(m)
        # We can use itertools.combinations
        for combo in itertools.combinations(range(m), 4):
            # Extract rows
            # rows = np.vstack([rel_vecs[i] for i in combo]) -> slow
            # We can just index the precomputed arrays
            
            # Build 4x4 matrix
            # We can do this in pure python/numpy indexing
            
            # Since m is small, this loop is okay if we don't do it too many times.
            # But we do this for EVERY candidate in EVERY restart.
            
            # Let's try to make this faster.
            # If we find a violation, return False.
            
            # Construct matrix
            # We can store rel_vecs and rel_sqs as 2D arrays.
            # rel_vecs is (m, 3), rel_sqs is (m,)
            
            # For a specific combo (i, j, k, l):
            # mat = np.empty((4,4))
            # mat[0] = np.append(rel_vecs[i], rel_sqs[i])
            # ...
            
            # This is still slow in Python.
            
            # Is there a mathematical shortcut?
            # 5 points are co-spherical iff the determinant of the 5x5 matrix is 0.
            # This doesn't help with the incremental check.
            
            # Let's assume the number of restarts is limited by time.
            # We will implement the check and see how fast it is.
            
            # Optimization: Sort points by some metric? No.
            
            # Let's just do it.
            
            # To speed up, we can check if the 4 points are coplanar?
            # If 4 points are coplanar, they don't define a unique sphere (infinite spheres pass through them? No, 4 points define a unique sphere unless they are coplanar, in which case there are infinite spheres, or they are collinear).
            # Wait. If 4 points are coplanar, do they define a sphere?
            # A sphere is determined by 5 points.
            # If 4 points are coplanar, they lie on the intersection of the plane and some sphere?
            # Actually, any 4 non-coplanar points define a unique sphere.
            # If 4 points are coplanar, they do NOT define a unique sphere.
            # However, the problem says "no 5 points lie on a common sphere".
            # If 4 points are coplanar, they lie on ANY sphere that intersects that plane in the circle passing through them (if such a circle exists).
            # Actually, 4 coplanar points lie on a circle (if concyclic) or not.
            # If they are not concyclic, no sphere passes through them?
            # No, a sphere intersects a plane in a circle (or point/empty).
            # So if 4 coplanar points are not concyclic, no sphere passes through all 4.
            # If they are concyclic, infinite spheres pass through them.
            # So, if we have 4 coplanar points in our set, and we add a 5th point, we need to check if that 5th point lies on ANY sphere passing through the 4.
            # This is complex.
            
            # The determinant check handles all cases.
            # If the 4 points are coplanar, the 4x4 determinant of (q-p, ...) might be 0?
            # Let's verify.
            # If q1..q4 are coplanar, they lie on a plane.
            # The condition for p, q1..q4 to be co-spherical is that the determinant is 0.
            # The determinant check is robust.
            
            # Let's proceed with the combination loop.
            
            # To improve speed, we can break early.
            
            # We'll use a helper to check a specific combo.
            
            # For now, let's just iterate.
            
            i, j, k, l = combo
            
            # Build the 4x4 matrix
            # We can use numpy indexing to get the rows
            # rows = np.vstack((rel_vecs[i], rel_vecs[j], rel_vecs[k], rel_vecs[l]))
            # This creates new arrays.
            
            # Better:
            # mat = np.empty((4, 4), dtype=np.float64)
            # mat[0, :3] = rel_vecs[i]
            # mat[0, 3] = rel_sqs[i]
            # ...
            
            # This is 4 * 4 assignments.
            
            # Let's try to vectorize the whole check for a given p?
            # We need to check C(m,4) determinants.
            # We can generate all C(m,4) indices, then index rel_vecs and rel_sqs, then stack.
            
            # If m is small (< 20), C(m,4) is small.
            # If m is 30, C(30,4) = 27k.
            # Generating 27k indices and stacking 27k * 4 * 4 floats is ~ 4MB.
            # np.linalg.det on (27000, 4, 4) might be fast.
            
            if m > 10:
                # Vectorized check
                combos = np.array(list(itertools.combinations(range(m), 4)))
                if len(combos) == 0:
                    return True
                
                # Index the relative vectors and squared norms
                # rel_vecs is (m, 3), rel_sqs is (m,)
                # combos is (C, 4)
                
                # We need to gather the rows.
                # v1 = rel_vecs[combos[:, 0]] -> (C, 3)
                # v2 = rel_vecs[combos[:, 1]] -> (C, 3)
                # ...
                
                v1 = rel_vecs[combos[:, 0]]
                v2 = rel_vecs[combos[:, 1]]
                v3 = rel_vecs[combos[:, 2]]
                v4 = rel_vecs[combos[:, 3]]
                
                s1 = rel_sqs[combos[:, 0]]
                s2 = rel_sqs[combos[:, 1]]
                s3 = rel_sqs[combos[:, 2]]
                s4 = rel_sqs[combos[:, 3]]
                
                # Stack into (C, 4, 4)
                # We can use np.stack or manual assignment
                # mat = np.empty((len(combos), 4, 4), dtype=np.float64)
                # mat[:, 0, :3] = v1
                # mat[:, 0, 3] = s1
                # ...
                
                # This is memory intensive but fast.
                
                mat = np.empty((len(combos), 4, 4), dtype=np.float64)
                mat[:, 0, 0:3] = v1
                mat[:, 0, 3] = s1
                mat[:, 1, 0:3] = v2
                mat[:, 1, 3] = s2
                mat[:, 2, 0:3] = v3
                mat[:, 2, 3] = s3
                mat[:, 3, 0:3] = v4
                mat[:, 3, 3] = s4
                
                dets = np.linalg.det(mat)
                return not np.any(np.abs(dets) < 0.5)
            else:
                # Small m, use loop
                for combo in itertools.combinations(range(m), 4):
                    i, j, k, l = combo
                    mat = np.empty((4, 4), dtype=np.float64)
                    mat[0, :3] = rel_vecs[i]
                    mat[0, 3] = rel_sqs[i]
                    mat[1, :3] = rel_vecs[j]
                    mat[1, 3] = rel_sqs[j]
                    mat[2, :3] = rel_vecs[k]
                    mat[2, 3] = rel_sqs[k]
                    mat[3, :3] = rel_vecs[l]
                    mat[3, 3] = rel_sqs[l]
                    if abs(np.linalg.det(mat)) < 0.5:
                        return False
                return True

    def greedy_fill(order):
        """
        Greedily add points from the given order.
        """
        S = []
        for p_idx in order:
            if can_add(np.array(S), p_idx):
                S.append(p_idx)
        return np.array(S, dtype=np.int64)

    def local_improve(S, deadline):
        """
        Try to improve the set S by adding points or swapping.
        """
        S_set = set(S.tolist())
        S_list = list(S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any point not in S
            for p_idx in range(N):
                if p_idx not in S_set:
                    if can_add(np.array(S_list), p_idx):
                        S_list.append(p_idx)
                        S_set.add(p_idx)
                        improved = True
                        break # Restart the loop to try adding more
            
            if not improved:
                # Try swaps
                # Randomize the order of points in S and points not in S
                # To avoid bias
                np.random.shuffle(S_list)
                outside = [i for i in range(N) if i not in S_set]
                np.random.shuffle(outside)
                
                for q_idx in S_list:
                    if time.time() > deadline:
                        return np.array(S_list)
                    for p_idx in outside:
                        # Check if we can swap q_idx with p_idx
                        # New set would be (S \ {q_idx}) U {p_idx}
                        # This is valid if p_idx is compatible with S \ {q_idx}
                        # Which means p_idx doesn't form a co-spherical set with any 4 points in S \ {q_idx}
                        
                        # We can check this by temporarily removing q_idx
                        # Or just check can_add on S \ {q_idx}
                        
                        # To save time, we can check if p_idx is "close" to q_idx? No.
                        
                        # Let's just do the check.
                        # S_temp = [x for x in S_list if x != q_idx]
                        # This is O(N) to filter.
                        
                        # We can create a mask
                        # But can_add expects an array of indices.
                        
                        # Let's create the temp array
                        S_temp = np.array([x for x in S_list if x != q_idx])
                        
                        if can_add(S_temp, p_idx):
                            # Perform swap
                            S_list.remove(q_idx)
                            S_list.append(p_idx)
                            S_set.remove(q_idx)
                            S_set.add(p_idx)
                            improved = True
                            break
                    if improved:
                        break
        
        return np.array(S_list, dtype=np.int64)

    # Main loop
    np.random.seed(42)
    best_S = np.array([], dtype=np.int64)
    deadline = time.time() + 590 # 10s margin
    
    while time.time() < deadline:
        # Generate a random order
        order = np.random.permutation(N)
        
        # Greedy fill
        S = greedy_fill(order)
        
        # Local improve
        S = local_improve(S, deadline)
        
        if len(S) > len(best_S):
            best_S = S
            # Optionally, print progress
            # print(f"New best: {len(best_S)}")

    # Convert indices to coordinates
    if len(best_S) == 0:
        return np.array([], dtype=np.int64).reshape(0, 3)
        
    return grid[best_S].astype(np.int64)

# EVOLVE-BLOCK-END
