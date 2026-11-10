# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin before 600s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)

    # Pre-compute combination indices for m=4 to m=60
    combos_cache = {}
    max_m = 60
    for m in range(4, max_m + 1):
        idx = np.arange(m)
        combos = list(itertools.combinations(idx, 4))
        if combos:
            combos_cache[m] = np.array(combos, dtype=np.int32)
        else:
            combos_cache[m] = np.empty((0, 4), dtype=np.int32)

    def can_add(arr_indices, p_idx):
        """
        arr_indices: list of indices of points already in the set.
        p_idx: index of candidate point to add.
        Returns True if adding p_idx keeps the set valid.
        """
        m = len(arr_indices)
        if m < 4:
            return True

        p = grid[p_idx]
        p_sq = grid_sq[p_idx]

        # Build matrix: rows[i] = (arr[i] - p, |arr[i]|^2 - |p|^2)
        arr_pts = grid[arr_indices]  # (m, 3)
        diffs = arr_pts.astype(np.float64) - p.astype(np.float64)  # (m, 3)
        sq_diffs = grid_sq[arr_indices] - p_sq  # (m,)
        rows = np.column_stack([diffs, sq_diffs])  # (m, 4)

        if m not in combos_cache:
            # Fallback for very large m (unlikely)
            idx = np.arange(m)
            combos = list(itertools.combinations(idx, 4))
            if not combos:
                return True
            combo_arr = np.array(combos)
        else:
            combo_arr = combos_cache[m]
            
        if combo_arr.shape[0] == 0:
            return True

        sub = rows[combo_arr]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def greedy_fill(order):
        """Greedy fill: add points in given order if valid."""
        arr_indices = []
        for idx in order:
            if time.time() > deadline - 30:
                break
            if can_add(arr_indices, idx):
                arr_indices.append(idx)
        return arr_indices

    def local_search(arr_indices, max_rounds=50):
        """
        Local Search: Try to add any valid point. If none, try swaps.
        """
        current = list(arr_indices)
        current_set = set(current)
        improved = True
        rounds = 0

        while improved and rounds < max_rounds and time.time() < deadline - 30:
            improved = False
            rounds += 1
            
            # Phase 1: Try to add any valid point
            # Check all candidates not in set
            candidates = [j for j in range(N) if j not in current_set]
            # Shuffle to avoid bias
            np.random.shuffle(candidates)
            
            for j in candidates:
                if can_add(current, j):
                    current.append(j)
                    current_set.add(j)
                    improved = True
                    break  # Restart scan after an add
            
            if improved:
                continue

            # Phase 2: Try swaps
            # If we couldn't add, try swapping one point for another
            # To speed up, we only try a limited number of swaps per round
            # We pick random points from current set and random candidates
            num_swaps_to_try = 200
            tried_swaps = 0
            while tried_swaps < num_swaps_to_try and time.time() < deadline - 30:
                i = np.random.choice(len(current))
                out_idx = current[i]
                candidates = [j for j in range(N) if j not in current_set]
                if not candidates:
                    break
                in_idx = np.random.choice(candidates)
                
                # Check if swap is valid
                remaining = [x for x in current if x != out_idx]
                if can_add(remaining, in_idx):
                    # Perform swap
                    current[i] = in_idx
                    current_set.remove(out_idx)
                    current_set.add(in_idx)
                    improved = True
                    break  # Restart scan after a swap
                tried_swaps += 1

        return current

    # --- Main search loop: Iterated Local Search ---
    rng = np.random.default_rng(42)
    best_set = []
    best_size = 0

    # Initial solution
    order = rng.permutation(N)
    current = greedy_fill(order)
    current = local_search(current, max_rounds=10)
    best_set = list(current)
    best_size = len(current)

    # ILS Loop
    while time.time() < deadline - 30:
        # Perturb best solution
        perturbed = list(best_set)
        if len(perturbed) > 0:
            # Remove 5-10% of points randomly
            num_remove = max(1, int(len(perturbed) * 0.1))
            if num_remove < len(perturbed):
                remove_indices = rng.choice(len(perturbed), size=num_remove, replace=False)
                # Sort to remove from end to avoid index shifting issues
                remove_indices = sorted(remove_indices, reverse=True)
                for idx in remove_indices:
                    perturbed.pop(idx)
        
        # Run Local Search on perturbed solution
        current = local_search(perturbed, max_rounds=20)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
            
        # Occasionally, try a completely new random restart if stuck?
        # For now, ILS is sufficient.

    # Convert best set to array
    if best_size > 0:
        result = np.array([grid[i] for i in best_set], dtype=np.int64)
    else:
        result = np.empty((0, 3), dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
