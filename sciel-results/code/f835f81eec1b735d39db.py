import numpy as np
import time

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280

    # 1D Sidon baseline
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    best_set = [(x, 0) for x in xs]
    best_size = len(best_set)

    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    num_points = len(all_points)
    all_pts64 = all_points.astype(np.int64)
    
    # Precompute distance matrix
    diff = all_pts64[:, np.newaxis, :] - all_pts64[np.newaxis, :, :]
    dist_matrix = np.sum(diff ** 2, axis=2).astype(np.int32)
    del diff

    # Vectorized Check Function
    def get_valid_candidates(p_idx_arr, curr_indices, dist_mat):
        """
        p_idx_arr: array of candidate indices (M,)
        curr_indices: list of indices in current set (K,)
        dist_mat: full distance matrix
        Returns: boolean mask of same length as p_idx_arr, True if valid.
        """
        if len(curr_indices) == 0:
            return np.ones(len(p_idx_arr), dtype=bool)
        
        curr_arr = np.array(curr_indices, dtype=np.int32)
        K = len(curr_arr)
        M = len(p_idx_arr)
        
        if M == 0:
            return np.array([], dtype=bool)

        # D[i, j] = dist(p_idx_arr[i], curr_arr[j])
        D = dist_mat[p_idx_arr[:, np.newaxis], curr_arr[np.newaxis, :]]
        
        # 1. Check isosceles at p (row uniqueness)
        # Sort rows and check for duplicates
        D_sorted = np.sort(D, axis=1)
        # If any adjacent are equal, it's invalid
        valid_p = np.all(np.diff(D_sorted, axis=1) != 0, axis=1)
        
        if not np.any(valid_p):
            return np.zeros(M, dtype=bool)
        
        # 2. Check isosceles at existing points q
        # For each q_j, we have distances to other points in S: V_j
        # We need D[:, j] NOT in V_j
        
        # Precompute V_j sets (as sorted arrays for isin?)
        # np.isin works with arrays
        valid_mask = valid_p.copy()
        
        for j in range(K):
            # Distances from q_j to all other points in S
            # curr_arr[j] is q_j
            # We need distances from curr_arr[j] to curr_arr excluding j
            others = np.delete(curr_arr, j)
            if len(others) == 0:
                continue
            V_j = dist_mat[curr_arr[j], others]
            
            # Check if D[:, j] is in V_j
            # D[:, j] is the column j of D
            # np.isin returns True if element is in V_j
            is_in = np.isin(D[:, j], V_j)
            valid_mask &= ~is_in
        
        return valid_mask

    def try_add_points(current_indices, remaining, dist_mat, max_adds=5):
        """Try to greedily add more points to the current set."""
        curr = list(current_indices)
        rem = remaining.copy()
        for _ in range(max_adds):
            if len(rem) == 0:
                break
            vm = get_valid_candidates(rem, curr, dist_mat)
            if not np.any(vm):
                break
            chosen = np.random.choice(rem[vm])
            curr.append(int(chosen))
            rem = np.setdiff1d(rem, np.array([chosen]))
        return curr, rem

    def local_search(indices, dist_mat, deadline):
        """Simple local search: try swaps to improve the set."""
        current = list(indices)
        curr_set = set(current)
        remaining = np.setdiff1d(np.arange(num_points), np.array(current))
        
        # Phase 1: Try to add points
        current, remaining = try_add_points(current, remaining, dist_mat, max_adds=10)
        
        # Phase 2: Try swap operations (remove 1, add 1-2)
        improved = True
        max_swaps = 50
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < deadline:
            improved = False
            swap_count += 1
            
            if len(current) < 2:
                break
            
            # Try a few random removals
            for _ in range(min(5, len(current))):
                if time.time() > deadline:
                    break
                
                rem_idx = np.random.randint(len(current))
                removed = current[rem_idx]
                new_current = current[:rem_idx] + current[rem_idx+1:]
                new_remaining = np.union1d(remaining, np.array([removed]))
                
                # Try to add up to 2 points
                temp_current, temp_remaining = try_add_points(new_current, new_remaining, dist_mat, max_adds=2)
                
                # Accept if we gained net points
                if len(temp_current) > len(current):
                    current = temp_current
                    remaining = temp_remaining
                    improved = True
                    break
                # Accept same-size with different composition for diversity
                elif len(temp_current) == len(current) and len(temp_current) > 0:
                    if np.random.random() < 0.2:
                        current = temp_current
                        remaining = temp_remaining
                        improved = True
                        break
        
        return current

    num_restarts = 250
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        current_indices = []
        remaining = np.arange(num_points)
        
        # Greedy selection
        while len(remaining) > 0 and time.time() < deadline:
            valid_mask = get_valid_candidates(remaining, current_indices, dist_matrix)
            
            if not np.any(valid_mask):
                break
            
            valid_indices = remaining[valid_mask]
            chosen_idx = np.random.choice(valid_indices)
            
            current_indices.append(chosen_idx)
            remaining = np.delete(remaining, np.where(remaining == chosen_idx)[0])

        # Local search to improve
        if time.time() < deadline - 10:
            current_indices = local_search(current_indices, dist_matrix, deadline - 10)

        if len(current_indices) > best_size:
            best_set = [(int(all_points[idx, 0]), int(all_points[idx, 1])) for idx in current_indices]
            best_size = len(current_indices)

    return best_set
