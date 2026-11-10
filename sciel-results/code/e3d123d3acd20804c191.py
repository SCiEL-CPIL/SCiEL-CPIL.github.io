import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 580  # Safety margin

    # Pre-compute squared norms for all grid points
    grid_sq_norms = (grid.astype(np.float64) ** 2).sum(axis=1)

    # Mapping from tuple to index
    grid_tuple_to_idx = {tuple(p): i for i, p in enumerate(grid)}

    def set_to_indices(s):
        return [grid_tuple_to_idx[tuple(p)] for p in s]

    def indices_to_set(idxs):
        return grid[list(idxs)]

    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if adding point p to arr keeps the set valid.
        arr: (m, 3) array of existing points.
        p: (3,) array of new point.
        Returns True if valid.
        """
        m = len(arr)
        if m < 4:
            return True

        # diff[i] = arr[i] - p
        diff = arr.astype(np.float64) - p.astype(np.float64)
        # sq_diff[i] = |arr[i]|^2 - |p|^2
        p_sq = (p.astype(np.float64) ** 2).sum()
        sq_diff = (arr.astype(np.float64) ** 2).sum(axis=1) - p_sq

        # Construct the (m, 4) matrix M where rows are (dx, dy, dz, dsq)
        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = sq_diff

        # Generate all C(m,4) index combinations
        combo_indices = list(itertools.combinations(range(m), 4))
        if not combo_indices:
            return True

        # Process in batches for early termination
        batch_size = 5000
        for start in range(0, len(combo_indices), batch_size):
            end = min(start + batch_size, len(combo_indices))
            idxs = np.array(combo_indices[start:end], dtype=np.int32)
            M_sub = M[idxs]
            dets = np.linalg.det(M_sub)
            if np.any(np.abs(dets) < 0.5):
                return False

        return True

    def greedy_build(order, existing_set_indices=None):
        """
        Greedily build a set from 'order' (list of indices) starting with 'existing_set_indices'.
        Returns set of indices.
        """
        if existing_set_indices is None:
            curr_indices = set()
        else:
            curr_indices = set(existing_set_indices)
        
        curr_set = indices_to_set(curr_indices)
        
        for idx in order:
            if time.time() > deadline:
                break
            if idx in curr_indices:
                continue
                
            p = grid[idx]
            if is_valid_addition(curr_set, p):
                curr_indices.add(idx)
                curr_set = np.vstack([curr_set, p]) if len(curr_set) > 0 else p[None, :]

        return curr_indices

    best_set_indices = set()
    best_size = 0

    np.random.seed(42)

    # Phase 1: Multiple Greedy Restarts
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        order = np.random.permutation(num_points_total)
        curr_indices = greedy_build(order)
        if len(curr_indices) > best_size:
            best_size = len(curr_indices)
            best_set_indices = curr_indices

    # Phase 2: Local Search
    # We will run local search on the best set found so far.
    # To be safe, we can run it on a few top sets.
    
    def local_search(start_indices, max_iters=5000):
        curr_indices = set(start_indices)
        rem_indices = set(range(num_points_total)) - curr_indices
        curr_set = indices_to_set(curr_indices)
        
        no_improve_count = 0
        max_no_improve = 100
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            
            # 1. Try Add
            if rem_indices:
                rem_list = list(rem_indices)
                idx = rem_list[np.random.randint(len(rem_list))]
                p = grid[idx]
                
                if is_valid_addition(curr_set, p):
                    curr_indices.add(idx)
                    rem_indices.remove(idx)
                    curr_set = np.vstack([curr_set, p]) if len(curr_set) > 0 else p[None, :]
                    improved = True
                    no_improve_count = 0
                    continue

            # 2. Try 1-Swap
            if curr_indices and rem_indices:
                in_list = list(curr_indices)
                out_list = list(rem_indices)
                
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    in_idx = in_list[np.random.randint(len(in_list))]
                    out_idx = out_list[np.random.randint(len(out_list))]
                    
                    # Check if swap is valid
                    reduced_indices = list(curr_indices - {in_idx})
                    reduced_set = indices_to_set(reduced_indices)
                    p_out = grid[out_idx]
                    
                    if is_valid_addition(reduced_set, p_out):
                        curr_indices.remove(in_idx)
                        curr_indices.add(out_idx)
                        rem_indices.add(in_idx)
                        rem_indices.remove(out_idx)
                        
                        # Update curr_set
                        mask = np.all(curr_set == grid[in_idx], axis=1)
                        curr_set[mask] = p_out
                        
                        improved = True
                        no_improve_count = 0
                        break
            
            # 3. Try 2-Swap
            if not improved and len(curr_indices) >= 2 and len(rem_indices) >= 2:
                in_list = list(curr_indices)
                out_list = list(rem_indices)
                
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    in_idxs = np.random.choice(in_list, 2, replace=False)
                    out_idxs = np.random.choice(out_list, 2, replace=False)
                    
                    in_set = set(in_idxs)
                    out_set = set(out_idxs)
                    
                    reduced_indices = list(curr_indices - in_set)
                    reduced_set = indices_to_set(reduced_indices)
                    
                    p1 = grid[out_idxs[0]]
                    p2 = grid[out_idxs[1]]
                    
                    if is_valid_addition(reduced_set, p1):
                        temp_set = np.vstack([reduced_set, p1]) if len(reduced_set) > 0 else p1[None, :]
                        if is_valid_addition(temp_set, p2):
                            curr_indices -= in_set
                            curr_indices |= out_set
                            rem_indices |= in_set
                            rem_indices -= out_set
                            
                            # Update curr_set
                            mask_in = np.zeros(len(curr_set), dtype=bool)
                            for idx in in_idxs:
                                mask_in |= np.all(curr_set == grid[idx], axis=1)
                            curr_set = curr_set[~mask_in]
                            curr_set = np.vstack([curr_set, p1, p2]) if len(curr_set) > 0 else np.vstack([p1[None,:], p2[None,:]])
                            
                            improved = True
                            no_improve_count = 0
                            break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        return curr_indices

    # Run local search
    curr_indices = local_search(best_set_indices, max_iters=10000)
    if len(curr_indices) > best_size:
        best_size = len(curr_indices)
        best_set_indices = curr_indices

    # Phase 3: Iterated Local Search
    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Perturb
        k = np.random.randint(2, 5)
        if len(best_set_indices) <= k:
            continue
            
        perturb_indices = np.random.choice(list(best_set_indices), k, replace=False)
        new_indices = best_set_indices - set(perturb_indices)
        new_rem = set(range(num_points_total)) - new_indices
        
        # Repair
        # Use greedy build starting from new_indices
        # Order: random permutation of new_rem
        order = np.random.permutation(list(new_rem))
        repaired_indices = greedy_build(order, existing_set_indices=new_indices)
        
        if len(repaired_indices) > best_size:
            best_size = len(repaired_indices)
            best_set_indices = repaired_indices

    return indices_to_set(best_set_indices).astype(np.int64)
