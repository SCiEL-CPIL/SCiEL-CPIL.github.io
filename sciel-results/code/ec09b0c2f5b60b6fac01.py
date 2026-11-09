import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 580  # Safety margin
    
    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if adding point p to arr keeps the set valid.
        Uses early termination for speed.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # diff[i] = arr[i] - p
        diff = arr - p
        # sq_diff[i] = |arr[i]|^2 - |p|^2
        sq_diff = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        
        # Construct the (m, 4) matrix M where rows are (dx, dy, dz, dsq)
        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = sq_diff
        
        # For efficiency, check combinations in batches and terminate early
        # Generate combinations in a way that allows early exit
        for combo in itertools.combinations(range(m), 4):
            sub = M[list(combo)]
            det_val = np.linalg.det(sub)
            if abs(det_val) < 0.5:
                return False
        
        return True

    def get_random_order():
        return np.random.permutation(num_points_total)

    def greedy_build(order):
        curr_set = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if is_valid_addition(curr_set, p):
                if len(curr_set) == 0:
                    curr_set = p[None, :]
                else:
                    curr_set = np.vstack([curr_set, p])
        return curr_set

    def greedy_build_sorted(order):
        """Greedy build with a bias towards points that are 'far' from existing ones."""
        curr_set = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if is_valid_addition(curr_set, p):
                if len(curr_set) == 0:
                    curr_set = p[None, :]
                else:
                    curr_set = np.vstack([curr_set, p])
        return curr_set

    best_set = np.empty((0, 3), dtype=np.int64)
    
    np.random.seed(42)
    
    # Phase 1: Multiple Greedy Restarts
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        order = get_random_order()
        curr_set = greedy_build(order)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Phase 2: Local Search
    current_set = best_set.copy()
    
    # Build a tuple-to-index map for the grid
    grid_tuple_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    
    current_indices = set()
    for p in current_set:
        current_indices.add(grid_tuple_to_idx[tuple(p)])
        
    all_indices = set(range(num_points_total))
    remaining_indices = list(all_indices - current_indices)
    
    no_improve_count = 0
    max_no_improve = 150
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to Add - check all remaining points (more thorough)
        if remaining_indices:
            # Check all remaining points, but in a random order for diversity
            shuffled_remaining = remaining_indices.copy()
            np.random.shuffle(shuffled_remaining)
            for idx in shuffled_remaining:
                if time.time() > deadline:
                    break
                p = grid[idx]
                if is_valid_addition(current_set, p):
                    # Add
                    if len(current_set) == 0:
                        current_set = p[None, :]
                    else:
                        current_set = np.vstack([current_set, p])
                    
                    current_indices.add(idx)
                    remaining_indices.remove(idx)
                    improved = True
                    no_improve_count = 0
                    break
        
        # 2. Try to Swap - more systematic approach
        if not improved and len(current_indices) > 0 and remaining_indices:
            # Try multiple swap attempts
            swaps_to_try = 50
            tried_swaps = set()
            for _ in range(swaps_to_try):
                if time.time() > deadline:
                    break
                
                # Pick random in
                in_idx_val = list(current_indices)[np.random.randint(len(current_indices))]
                # Pick random out
                out_idx_val = remaining_indices[np.random.randint(len(remaining_indices))]
                
                swap_key = (in_idx_val, out_idx_val)
                if swap_key in tried_swaps:
                    continue
                tried_swaps.add(swap_key)
                
                p_in = grid[in_idx_val]
                p_out = grid[out_idx_val]
                
                # Find index of p_in in current_set
                in_pos = None
                for i in range(len(current_set)):
                    if np.array_equal(current_set[i], p_in):
                        in_pos = i
                        break
                
                if in_pos is None:
                    continue
                    
                reduced_mask = np.ones(len(current_set), dtype=bool)
                reduced_mask[in_pos] = False
                reduced_set = current_set[reduced_mask]
                
                if is_valid_addition(reduced_set, p_out):
                    # Perform swap
                    current_set = reduced_set
                    if len(current_set) == 0:
                        current_set = p_out[None, :]
                    else:
                        current_set = np.vstack([current_set, p_out])
                    
                    current_indices.remove(in_idx_val)
                    current_indices.add(out_idx_val)
                    remaining_indices.remove(out_idx_val)
                    remaining_indices.append(in_idx_val)
                    
                    improved = True
                    no_improve_count = 0
                    break
        
        # 3. Try 2-for-1 or 1-for-2 moves (remove one, add two)
        if not improved and len(current_indices) > 2 and len(remaining_indices) >= 2:
            moves_to_try = 30
            for _ in range(moves_to_try):
                if time.time() > deadline:
                    break
                
                # Remove one point
                in_idx_val = list(current_indices)[np.random.randint(len(current_indices))]
                p_in = grid[in_idx_val]
                
                # Find and remove
                in_pos = None
                for i in range(len(current_set)):
                    if np.array_equal(current_set[i], p_in):
                        in_pos = i
                        break
                
                if in_pos is None:
                    continue
                    
                reduced_mask = np.ones(len(current_set), dtype=bool)
                reduced_mask[in_pos] = False
                reduced_set = current_set[reduced_mask]
                
                # Try to add two points from remaining
                if len(remaining_indices) < 2:
                    continue
                    
                idx1, idx2 = np.random.choice(remaining_indices, 2, replace=False)
                p1 = grid[idx1]
                p2 = grid[idx2]
                
                # Check if both can be added
                if is_valid_addition(reduced_set, p1):
                    temp_set = np.vstack([reduced_set, p1[None, :]]) if len(reduced_set) > 0 else p1[None, :]
                    if is_valid_addition(temp_set, p2):
                        # Perform the move
                        current_set = temp_set
                        current_set = np.vstack([current_set, p2[None, :]]) if len(current_set) > 0 else p2[None, :]
                        
                        current_indices.remove(in_idx_val)
                        current_indices.add(idx1)
                        current_indices.add(idx2)
                        remaining_indices.remove(idx1)
                        remaining_indices.remove(idx2)
                        remaining_indices.append(in_idx_val)
                        
                        improved = True
                        no_improve_count = 0
                        break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Kick - remove several random points to escape local optimum
                if len(current_indices) > 10:
                    kick_size = min(8, len(current_indices) // 3)
                    # Remove random points
                    kick_idxs = np.random.choice(list(current_indices), kick_size, replace=False)
                    for idx in kick_idxs:
                        p = grid[idx]
                        # Remove from current_set
                        mask = np.ones(len(current_set), dtype=bool)
                        for i in range(len(current_set)):
                            if np.array_equal(current_set[i], p):
                                mask[i] = False
                                break
                        current_set = current_set[mask]
                        
                        current_indices.remove(idx)
                        remaining_indices.append(idx)
                    no_improve_count = 0
                    max_no_improve = 250

    return current_set.astype(np.int64)
