import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    def can_add_fast(arr_idx, p_idx):
        m = len(arr_idx)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        arr_pts = grid[arr_idx]
        arr_sq = grid_sq[arr_idx]
        
        diff = arr_pts - p
        sq_diff = arr_sq - p_sq
        
        # Check all 4-combinations with early termination
        combo_list = list(itertools.combinations(range(m), 4))
        total = len(combo_list)
        batch_size = 3000
        for start in range(0, total, batch_size):
            batch = combo_list[start:start+batch_size]
            if not batch:
                break
            combos = np.array(batch, dtype=np.int64)
            sub_diff = diff[combos]
            sub_sq = sq_diff[combos]
            mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)
            dets = np.linalg.det(mat)
            if np.any(np.abs(dets) < 0.5):
                return False
        return True

    def can_add_multiple(arr_idx, p_indices):
        """Check if multiple points can be added to arr_idx. Returns list of addable indices."""
        m = len(arr_idx)
        if m < 4:
            return list(p_indices)
        
        arr_pts = grid[arr_idx]
        arr_sq = grid_sq[arr_idx]
        addable = []
        
        for p_idx in p_indices:
            p = grid[p_idx]
            p_sq = grid_sq[p_idx]
            
            diff = arr_pts - p
            sq_diff = arr_sq - p_sq
            
            combo_list = list(itertools.combinations(range(m), 4))
            total = len(combo_list)
            batch_size = 3000
            valid = True
            for start in range(0, total, batch_size):
                batch = combo_list[start:start+batch_size]
                if not batch:
                    break
                combos = np.array(batch, dtype=np.int64)
                sub_diff = diff[combos]
                sub_sq = sq_diff[combos]
                mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)
                dets = np.linalg.det(mat)
                if np.any(np.abs(dets) < 0.5):
                    valid = False
                    break
            
            if valid:
                addable.append(p_idx)
        
        return addable

    def greedy_fill(order):
        selected = []
        for idx in order:
            if can_add_fast(np.array(selected), idx):
                selected.append(idx)
        return np.array(selected)
    
    def deterministic_polish(selected_arr, time_budget=5.0):
        """Systematically try to add remaining points in multiple orderings."""
        current = list(selected_arr)
        current_set = set(current)
        start_time = time.time()
        
        # Try multiple orderings
        orderings = [
            list(range(N)),  # natural order
            list(range(N-1, -1, -1)),  # reverse
            sorted(range(N), key=lambda i: grid[i].sum()),  # by coordinate sum
            sorted(range(N), key=lambda i: np.linalg.norm(grid[i] - 3.5)),  # by distance from center
        ]
        
        for order in orderings:
            if time.time() - start_time > time_budget:
                break
            for idx in order:
                if idx in current_set:
                    continue
                if can_add_fast(np.array(current), idx):
                    current.append(idx)
                    current_set.add(idx)
        
        return np.array(current)

    def local_search(selected_arr, time_limit):
        current = list(selected_arr)
        current_set = set(current)
        start_time = time.time()
        rng = np.random.default_rng(42)
        
        while time.time() < time_limit:
            improved = False
            
            # Try to add points in sorted order (deterministic first)
            for idx in range(N):
                if idx in current_set:
                    continue
                if can_add_fast(np.array(current), idx):
                    current.append(idx)
                    current_set.add(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one point, try to add a different one
            if len(current) > 0:
                # Try multiple random removals before giving up
                for _ in range(15):
                    if time.time() >= time_limit:
                        break
                    out_idx_pos = rng.integers(len(current))
                    out_idx = current[out_idx_pos]
                    
                    candidates = [i for i in range(N) if i not in current_set]
                    if not candidates:
                        break
                    rng.shuffle(candidates)
                    
                    swapped = False
                    for in_idx in candidates[:400]:  # Limit candidates to speed up
                        temp_current = current.copy()
                        temp_current.pop(out_idx_pos)
                        if can_add_fast(np.array(temp_current), in_idx):
                            current[out_idx_pos] = in_idx
                            current_set.remove(out_idx)
                            current_set.add(in_idx)
                            swapped = True
                            improved = True
                            break
                    
                    if swapped:
                        break
                    elif not improved:
                        # Try another random removal
                        continue
            
            if not improved:
                # No improvement possible, break
                break

        return np.array(current)

    def final_polish(selected_arr, time_limit):
        """Aggressive final polish with multiple strategies."""
        current = list(selected_arr)
        current_set = set(current)
        start_time = time.time()
        
        # Strategy 1: Try all points in natural order
        for idx in range(N):
            if time.time() >= time_limit:
                break
            if idx in current_set:
                continue
            if can_add_fast(np.array(current), idx):
                current.append(idx)
                current_set.add(idx)
        
        # Strategy 2: Try points sorted by various criteria
        if time.time() < time_limit - 2:
            criteria = [
                lambda i: grid[i].sum(),
                lambda i: np.linalg.norm(grid[i] - 3.5),
                lambda i: -grid[i].sum(),
                lambda i: np.linalg.norm(grid[i] - np.array([0,0,0])),
                lambda i: np.linalg.norm(grid[i] - np.array([7,7,7])),
            ]
            for crit in criteria:
                if time.time() >= time_limit:
                    break
                order = sorted(range(N), key=crit)
                for idx in order:
                    if time.time() >= time_limit:
                        break
                    if idx in current_set:
                        continue
                    if can_add_fast(np.array(current), idx):
                        current.append(idx)
                        current_set.add(idx)
        
        # Strategy 3: Local search with more iterations
        if time.time() < time_limit - 5:
            current = local_search(np.array(current), time.time() + min(10, time_limit - time.time()))
        
        return np.array(current)

    best_size = 0
    best_arr = np.array([], dtype=np.int64)
    
    start_time = time.time()
    deadline = start_time + 630 - 30  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Quick greedy fills with various orderings
    while time.time() < deadline - 120:
        r = rng.random()
        if r < 0.25:
            order = rng.permutation(N)
        elif r < 0.5:
            center = np.array([3.5, 3.5, 3.5])
            dists = np.linalg.norm(grid - center, axis=1)
            noise = rng.random(N) * 3.0
            order = np.argsort(dists + noise)
        elif r < 0.75:
            sums = grid.sum(axis=1)
            noise = rng.random(N) * 3.0
            order = np.argsort(sums + noise)
        else:
            # Random weighted order
            weights = rng.random(N)
            order = np.argsort(weights)
            
        current = greedy_fill(order)
        current = deterministic_polish(current, time_budget=3.0)
        
        if len(current) > len(best_arr):
            best_arr = current.copy()
            best_size = len(current)
    
    # Phase 2: Local search on best solutions
    if time.time() < deadline - 60:
        current = local_search(best_arr, time.time() + min(30, deadline - time.time() - 60))
        current = deterministic_polish(current, time_budget=5.0)
        if len(current) > len(best_arr):
            best_arr = current.copy()
            best_size = len(current)
    
    # Phase 3: More local search rounds
    while time.time() < deadline - 30:
        current = local_search(best_arr, time.time() + min(15, deadline - time.time() - 30))
        current = deterministic_polish(current, time_budget=3.0)
        if len(current) > len(best_arr):
            best_arr = current.copy()
            best_size = len(current)
        else:
            # Try a fresh local search from a slightly perturbed state
            if len(best_arr) > 10:
                perturbed = list(best_arr)
                # Remove a few random points and try to refill
                for _ in range(3):
                    if time.time() >= deadline - 30:
                        break
                    pos = rng.integers(len(perturbed))
                    removed = perturbed.pop(pos)
                    # Try to add it back or something else
                    for idx in range(N):
                        if idx not in set(perturbed):
                            if can_add_fast(np.array(perturbed), idx):
                                perturbed.append(idx)
                                break
                current = local_search(np.array(perturbed), time.time() + min(10, deadline - time.time() - 30))
                current = deterministic_polish(current, time_budget=3.0)
                if len(current) > len(best_arr):
                    best_arr = current.copy()
                    best_size = len(current)
    
    # Final aggressive polish
    if len(best_arr) > 0 and time.time() < deadline - 5:
        best_arr = final_polish(best_arr, deadline)
    
    return grid[best_arr]
