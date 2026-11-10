# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    n x n x n integer grid {0,...,n-1}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points).

    Here n = 8. The objective is to MAXIMIZE the number of selected points.

    Construction: run multiple seeded random-order greedy passes over the grid,
    adding a point whenever it keeps the set sphere-free, then apply a local
    improvement (swap) pass. Keep the largest set found. Deterministic via
    np.random.seed(0). Includes a deadline guard to stay within 600s.

    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, n-1], no five of which are co-spherical.
    """
    n = 8
    deadline = time.time() + 585  # safety margin before 600s limit

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq_norms = (grid ** 2).sum(axis=1)  # precompute |p|^2 for all grid points

    def can_add(arr: np.ndarray, p: np.ndarray, arr_sq_norms: np.ndarray) -> bool:
        """Check if adding point p to arr keeps the set sphere-free."""
        m = len(arr)
        if m < 4:
            return True
        # For large sets, we can early-terminate if too many combos would be needed
        # C(m,4) grows fast; for m=50, C(50,4)=230300 which is manageable
        # For m=60, C(60,4)=487635 - still ok
        # For m=70, C(70,4)=916895 - getting expensive
        # Use a threshold: if m > 65, sample combos instead of all
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq_norms - (p ** 2).sum()
        
        if m <= 65:
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        else:
            # Sample a subset of combinations for large sets
            # Still need to be thorough - use all if not too large
            # Actually let's just do all - numpy det is fast enough
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        return True

    def can_swap(arr: np.ndarray, remove_idx: int, p: np.ndarray, 
                 arr_sq_norms: np.ndarray) -> bool:
        """Check if removing arr[remove_idx] and adding p keeps the set sphere-free."""
        m = len(arr) - 1
        if m < 4:
            return True
        
        if remove_idx == 0:
            new_arr = arr[1:]
            new_sq = np.delete(arr_sq_norms, remove_idx)
        elif remove_idx == m:
            new_arr = arr[:m]
            new_sq = arr_sq_norms[:m]
        else:
            new_arr = np.concatenate([arr[:remove_idx], arr[remove_idx+1:]])
            new_sq = np.concatenate([arr_sq_norms[:remove_idx], arr_sq_norms[remove_idx+1:]])
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = new_arr - p
        w[:, 3] = new_sq - (p ** 2).sum()
        
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def greedy_pass(order: np.ndarray) -> np.ndarray:
        """Run a greedy pass over the given order of points."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq_norms = np.empty(0, dtype=np.float64)
        for idx in order:
            p = grid[idx]
            p_sq = grid_sq_norms[idx]
            if can_add(arr, p, arr_sq_norms):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq_norms = p_sq[None, :]
                else:
                    arr = np.vstack([arr, p])
                    arr_sq_norms = np.append(arr_sq_norms, p_sq)
        return arr

    def local_improve(arr: np.ndarray, time_budget: float) -> np.ndarray:
        """Try to improve arr by swapping one point for another, or removing a point
        and trying to add a different one."""
        arr = arr.copy()
        arr_sq_norms = (arr ** 2).sum(axis=1).astype(np.float64)
        start_time = time.time()
        selected_set = set(map(tuple, arr.tolist()))
        
        # Precompute grid points not in selected set
        grid_not_selected = [i for i in range(len(grid)) if tuple(grid[i]) not in selected_set]
        
        max_candidates_per_swap = 200  # Limit candidates tried per swap attempt
        
        while time.time() - start_time < time_budget:
            improved = False
            # Shuffle the order we try removals to avoid bias
            indices = np.random.permutation(len(arr))
            for i in indices:
                if time.time() - start_time > time_budget:
                    break
                removed = arr[i]
                # Try a random subset of candidates
                if len(grid_not_selected) > max_candidates_per_swap:
                    candidate_indices = np.random.choice(len(grid_not_selected), 
                                                         size=max_candidates_per_swap, 
                                                         replace=False)
                else:
                    candidate_indices = np.random.permutation(len(grid_not_selected))
                
                found = False
                for ci in candidate_indices:
                    gi = grid_not_selected[ci]
                    p = grid[gi]
                    p_sq = grid_sq_norms[gi]
                    if can_swap(arr, i, p, arr_sq_norms):
                        # Perform swap
                        new_arr = arr.copy()
                        new_arr[i] = p
                        arr = new_arr
                        arr_sq_norms = (arr ** 2).sum(axis=1).astype(np.float64)
                        selected_set.remove(tuple(removed))
                        selected_set.add(tuple(p))
                        # Remove gi from candidates list
                        grid_not_selected[ci] = grid_not_selected[-1]
                        grid_not_selected.pop()
                        improved = True
                        found = True
                        break
                if found:
                    break
            if not improved:
                # Try a "remove and refill" strategy: remove a random point,
                # then try to greedily add as many new points as possible
                if len(arr) > 10:
                    remove_idx = np.random.randint(len(arr))
                    removed = arr[remove_idx]
                    # Remove the point
                    if remove_idx == 0:
                        new_arr = arr[1:]
                    elif remove_idx == len(arr) - 1:
                        new_arr = arr[:-1]
                    else:
                        new_arr = np.concatenate([arr[:remove_idx], arr[remove_idx+1:]])
                    
                    # Try to add points back in random order
                    new_selected_set = set(map(tuple, new_arr.tolist()))
                    new_grid_not_selected = [i for i in range(len(grid)) 
                                            if tuple(grid[i]) not in new_selected_set]
                    random_order = np.random.permutation(len(new_grid_not_selected))
                    
                    trial_arr = new_arr.copy()
                    trial_sq = (trial_arr ** 2).sum(axis=1).astype(np.float64)
                    added_count = 0
                    
                    for ri in random_order:
                        if time.time() - start_time > time_budget:
                            break
                        gi = new_grid_not_selected[ri]
                        p = grid[gi]
                        p_sq = grid_sq_norms[gi]
                        if can_add(trial_arr, p, trial_sq):
                            if len(trial_arr) == 0:
                                trial_arr = p[None, :]
                                trial_sq = p_sq[None, :]
                            else:
                                trial_arr = np.vstack([trial_arr, p])
                                trial_sq = np.append(trial_sq, p_sq)
                            added_count += 1
                    
                    # Only accept if we gained at least 1 net point
                    if len(trial_arr) > len(arr):
                        arr = trial_arr
                        arr_sq_norms = trial_sq
                        selected_set = set(map(tuple, arr.tolist()))
                        grid_not_selected = [i for i in range(len(grid)) 
                                            if tuple(grid[i]) not in selected_set]
                        improved = True
                if not improved:
                    break
        return arr

    np.random.seed(0)
    best = None
    best_len = 0
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        order = np.random.permutation(len(grid))
        arr = greedy_pass(order)
        
        if best is None or len(arr) > best_len:
            best = arr.copy()
            best_len = len(arr)
    
    # Apply local improvement to the best solution
    if best is not None and time.time() < deadline:
        remaining = deadline - time.time() - 10
        if remaining > 5:
            improved = local_improve(best, remaining)
            if len(improved) > best_len:
                best = improved
                best_len = len(improved)
    
    return best.astype(np.int64)


# EVOLVE-BLOCK-END
