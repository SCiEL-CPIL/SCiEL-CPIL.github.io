def construct():
    import time
    import random
    import numpy as np
    
    deadline = time.time() + 290
    N = 64
    num_points = N * N
    
    # 1. Precompute all points and distances
    points = [(x, y) for y in range(N) for x in range(N)]
    
    # Distance matrix: dist_matrix[i, j] = squared distance between points[i] and points[j]
    # Using numpy for speed
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # dist_matrix[i, j] = (xs[i]-xs[j])^2 + (ys[i]-ys[j])^2
    # This is a 4096x4096 matrix.
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    
    # Precompute distance lists for each point to speed up local search
    # dist_lists[i] is a list of (j, dist_matrix[i, j]) for all j != i
    # This might be memory heavy (4096 * 4095 tuples), but let's see.
    # Actually, we can just use the matrix directly in the inner loop if optimized.
    # But iterating over a list of tuples is faster than indexing a 2D numpy array in Python.
    # Let's create a list of arrays: dist_lists[i] = dist_matrix[i] (excluding self)
    # We need to skip i itself.
    
    # Helper to check if a point can be added
    def can_add(idx, S, dist_sets):
        """
        Check if point idx can be added to set S.
        S is a list of indices.
        dist_sets is a dict: idx -> set of distances to other points in S.
        """
        # Distances from idx to all points in S
        # We need to check:
        # 1. dist(idx, q) is unique for all q in S
        # 2. dist(idx, q) is not in dist_sets[q] for all q in S
        
        # Optimization: Iterate through S
        # If S is large, this is O(|S|).
        # We can early exit.
        
        seen_dists = set()
        for q_idx in S:
            d = dist_matrix[idx, q_idx]
            if d in seen_dists:
                return False
            if d in dist_sets[q_idx]:
                return False
            seen_dists.add(d)
        return True

    def add_point(idx, S, dist_sets):
        S.append(idx)
        new_dists = set()
        for q_idx in S[:-1]:
            d = dist_matrix[idx, q_idx]
            new_dists.add(d)
            dist_sets[q_idx].add(d)
        dist_sets[idx] = new_dists

    def remove_point(idx, S, dist_sets):
        # Remove idx from S
        S.remove(idx)
        # Remove distances from idx to other points in S from their sets
        for q_idx in S:
            d = dist_matrix[idx, q_idx]
            dist_sets[q_idx].discard(d)
        # Clean up idx's set
        if idx in dist_sets:
            del dist_sets[idx]

    def greedy_build(rng):
        S = []
        dist_sets = {}
        candidates = list(range(num_points))
        rng.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline - 5:
                break
            if can_add(idx, S, dist_sets):
                add_point(idx, S, dist_sets)
        return S, dist_sets

    def local_search(S, dist_sets, rng):
        """
        Improve the set using swaps and additions.
        """
        current_S = list(S)
        current_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
        best_S = list(current_S)
        best_size = len(current_S)
        
        # Track time
        start_time = time.time()
        
        # To speed up candidate selection, we can precompute a list of indices not in S
        # But S changes.
        # Instead, we just pick random indices and check if they are in S.
        
        while time.time() < deadline - 2:
            # 1. Try to add a point
            # Sample a few random candidates
            added = False
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                cand_idx = rng.randint(0, num_points - 1)
                if cand_idx in current_dist_sets:
                    continue
                if can_add(cand_idx, current_S, current_dist_sets):
                    add_point(cand_idx, current_S, current_dist_sets)
                    added = True
                    if len(current_S) > best_size:
                        best_S = list(current_S)
                        best_size = len(current_S)
                    break
            
            if added:
                continue
                
            # 2. If no point added, perform a "kick" (swap) to escape local optimum
            # Remove a random point
            if not current_S:
                break
            rem_idx = current_S[rng.randint(0, len(current_S) - 1)]
            remove_point(rem_idx, current_S, current_dist_sets)
            
            # Try to add a random point
            # We want to restore the size, so we try to add a point that fits
            swap_success = False
            # Try multiple candidates to find a valid replacement
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                add_idx = rng.randint(0, num_points - 1)
                if add_idx == rem_idx:
                    continue
                if add_idx in current_dist_sets:
                    continue
                if can_add(add_idx, current_S, current_dist_sets):
                    add_point(add_idx, current_S, current_dist_sets)
                    swap_success = True
                    break
            
            # If swap failed, put rem_idx back
            if not swap_success:
                # Re-add rem_idx
                # We need to check if it can be added back
                # It should be, since we just removed it from a valid set
                # But to be safe, we can just re-insert it into the data structures
                # Actually, remove_point cleaned up the sets.
                # We can just call add_point(rem_idx)
                add_point(rem_idx, current_S, current_dist_sets)
                
        return best_S

    # Main loop
    best_S_global = []
    best_dist_sets_global = {}
    rng = random.Random(42)
    
    # Run multiple greedy restarts
    for i in range(100):
        if time.time() > deadline - 10:
            break
        
        S, dist_sets = greedy_build(rng)
        
        if len(S) > len(best_S_global):
            best_S_global = S
            best_dist_sets_global = dist_sets
            # Run local search on the best found so far
            if time.time() < deadline - 5:
                improved_S = local_search(S, dist_sets, rng)
                if len(improved_S) > len(best_S_global):
                    best_S_global = improved_S
                    # Reconstruct dist_sets for best_S_global if needed for further improvement
                    # But we return points, so we just need the indices.
                    # However, if we want to do more local search, we need the sets.
                    # Let's reconstruct them.
                    # This is O(k^2).
                    new_dist_sets = {}
                    for idx in best_S_global:
                        new_dist_sets[idx] = set()
                    for i_idx in range(len(best_S_global)):
                        for j_idx in range(i_idx + 1, len(best_S_global)):
                            idx1 = best_S_global[i_idx]
                            idx2 = best_S_global[j_idx]
                            d = dist_matrix[idx1, idx2]
                            new_dist_sets[idx1].add(d)
                            new_dist_sets[idx2].add(d)
                    best_dist_sets_global = new_dist_sets

    # To ensure we return a valid set, we should verify or just trust the algorithm.
    # The algorithm maintains validity.
    
    # Convert indices back to points
    result_points = [points[i] for i in best_S_global]
    
    return result_points
