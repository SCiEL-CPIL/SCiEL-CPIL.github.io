import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    # Precompute all grid points
    xs = np.arange(N)
    ys = np.arange(N)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    all_points = np.column_stack([XX.ravel(), YY.ravel()])  # shape (4096, 2)
    n_pts = N * N
    
    # Precompute squared distances between all pairs
    # 4096 x 4096 uint16 matrix
    # Max dist is (N-1)^2 + (N-1)^2 = 2 * 63^2 = 7938. Fits in uint16.
    # Using int32 for safety in intermediate calc, then cast.
    dx = all_points[:, 0:1].astype(np.int32) - all_points[1:, 0].astype(np.int32)
    dy = all_points[:, 1:2].astype(np.int32) - all_points[1:, 1].astype(np.int32)
    # This creates (4095, 4096)
    d_mat = dx * dx + dy * dy
    
    # We want a full (4096, 4096) matrix for O(1) lookup.
    # dist_mat[i, j] = d_mat[i, j-i-1] if i < j
    # dist_mat[j, i] = d_mat[i, j-i-1] if i < j
    dist_mat = np.zeros((n_pts, n_pts), dtype=np.uint16)
    for i in range(n_pts - 1):
        # d_mat[i] corresponds to distances from i to i+1 ... n_pts-1
        # We need to place these into dist_mat[i, i+1:] and dist_mat[i+1:, i]
        dist_mat[i, i+1:] = d_mat[i]
        dist_mat[i+1:, i] = d_mat[i]
    # Diagonal is 0, already initialized to 0.
    
    best_set = []
    best_size = 0
    
    def try_add(p_idx, S, dist_sets):
        """
        Try to add point p_idx to the set S.
        S: list of point indices.
        dist_sets: list of sets of distances. dist_sets[i] contains d(S[i], S[j]) for j != i.
        Returns (success, new_S, new_dist_sets)
        """
        k = len(S)
        if k == 0:
            return True, [p_idx], [set()]
        
        # Get distances from p_idx to all points in S
        # dist_mat[p_idx, S] gives an array of distances
        dists = dist_mat[p_idx, S]
        
        # Check 1: All distances must be distinct
        if len(np.unique(dists)) != k:
            return False, None, None
        
        # Check 2: For each existing point i, the distance d_i must not be in dist_sets[i]
        # We can check this by iterating or using numpy if we convert sets to arrays?
        # Sets are better for membership.
        for i in range(k):
            if dists[i] in dist_sets[i]:
                return False, None, None
        
        # Valid, update
        new_S = S + [p_idx]
        new_dist_sets = [set(ds) for ds in dist_sets] # Copy sets
        
        # Create new set for the new point
        new_set_for_p = set(dists)
        new_dist_sets.append(new_set_for_p)
        
        # Update existing sets
        for i in range(k):
            new_dist_sets[i].add(dists[i])
            
        return True, new_S, new_dist_sets

    def greedy_from_order(order, S, dist_sets, time_limit):
        """Try to add points from order to the set."""
        for idx in order:
            if time.time() > time_limit:
                break
            success, new_S, new_dist_sets = try_add(idx, S, dist_sets)
            if success:
                S = new_S
                dist_sets = new_dist_sets
        return S, dist_sets

    def local_search(S, dist_sets, time_limit):
        """
        Try to improve by removing a point and trying to add others.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(S)
            if k == 0:
                break
            
            # Try removing each point
            # We iterate through a random permutation of indices to try
            indices_to_try = list(range(k))
            random.shuffle(indices_to_try)
            
            for remove_idx in indices_to_try:
                if time.time() > time_limit:
                    break
                
                # Remove point at remove_idx
                # Rebuild S and dist_sets for the remaining points
                rem_S = [S[i] for i in range(k) if i != remove_idx]
                
                # Rebuild dist_sets from scratch for rem_S
                # This is O(k^2) which is fine for small k
                rem_k = len(rem_S)
                rem_dist_sets = [set() for _ in range(rem_k)]
                for i in range(rem_k):
                    for j in range(i + 1, rem_k):
                        d = dist_mat[rem_S[i], rem_S[j]]
                        rem_dist_sets[i].add(d)
                        rem_dist_sets[j].add(d)
                        
                # Try to add points from the complement
                # We need a list of candidates not in rem_S
                # Creating a boolean mask of size 4096 is fast
                in_set = np.zeros(n_pts, dtype=bool)
                for pt in rem_S:
                    in_set[pt] = True
                candidates = np.where(~in_set)[0]
                random.shuffle(candidates)
                
                added_count = 0
                new_rem_S = rem_S
                new_rem_dist_sets = rem_dist_sets
                
                # Try to add as many as possible, up to 2 (to net gain 1)
                # Actually, we can try to add all possible to maximize growth, 
                # but stopping at 2 is a good heuristic for "swap" improvement.
                # Let's try to add up to 3 to be safe? No, 2 is enough to replace 1.
                # If we add 1, size is same. If we add 2, size increases by 1.
                
                for c_idx in candidates:
                    if added_count >= 2:
                        break
                    success, new_S_try, new_ds_try = try_add(c_idx, new_rem_S, new_rem_dist_sets)
                    if success:
                        new_rem_S = new_S_try
                        new_rem_dist_sets = new_ds_try
                        added_count += 1
                
                # If we added more than we removed (1), we improved
                if added_count > 1:
                    S = new_rem_S
                    dist_sets = new_rem_dist_sets
                    improved = True
                    break
        
        return S, dist_sets

    num_restarts = 10000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        S = []
        dist_sets = []
        
        # Strategy: mix of random and structured initializations
        if restart % 5 == 0:
            order = np.random.permutation(n_pts)
        elif restart % 5 == 1:
            order = np.arange(n_pts)
        elif restart % 5 == 2:
            order = np.arange(n_pts).reshape(N, N).T.ravel()
        elif restart % 5 == 3:
            # Diagonal order
            order = []
            for d in range(2*N - 1):
                for x in range(max(0, d-N+1), min(N, d+1)):
                    y = d - x
                    if 0 <= y < N:
                        order.append(x * N + y)
            order = np.array(order)
        else:
            order = np.random.permutation(n_pts)
        
        S, dist_sets = greedy_from_order(order, S, dist_sets, deadline)
        
        # Second pass with different order to fill gaps
        if time.time() < deadline - 5:
            order2 = np.random.permutation(n_pts)
            S, dist_sets = greedy_from_order(order2, S, dist_sets, deadline)
        
        # Local search
        if time.time() < deadline - 10 and len(S) >= 5:
            S, dist_sets = local_search(S, dist_sets, deadline - 5)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = [(int(all_points[pt][0]), int(all_points[pt][1])) for pt in S]
            
    # Final polish on the best set found
    if best_size > 0 and time.time() < deadline - 5:
        # Reconstruct S and dist_sets from best_set
        # Map (x, y) back to index
        best_indices = []
        for x, y in best_set:
            best_indices.append(x * N + y)
        
        k = len(best_indices)
        dist_sets = [set() for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                d = dist_mat[best_indices[i], best_indices[j]]
                dist_sets[i].add(d)
                dist_sets[j].add(d)
        
        polished_S, polished_dist_sets = local_search(best_indices, dist_sets, deadline)
        
        if len(polished_S) > best_size:
            best_size = len(polished_S)
            best_set = [(int(all_points[pt][0]), int(all_points[pt][1])) for pt in polished_S]

    return best_set
