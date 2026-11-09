# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points as possible from the 8x8x8 integer grid {0,...,7}^3
    such that no five selected points are co-spherical (or co-planar).

    Strategy:
      1. Precompute per-point squared norms.
      2. Run many randomized greedy passes (with a deadline guard).
         - Each pass picks points in a random order and adds a point iff it
           does not create a 5-tuple co-spherical with any 4 already chosen.
      3. After each greedy pass, attempt a local-improvement pass:
         - For each point skipped during the greedy phase, try re-inserting it
           (checking against the current set).
         - Also try swapping: remove a point from the set and add two skipped
           points if valid (limited attempts for time).
      4. Keep the best set found before the deadline.

    Returns:
        np.ndarray of shape (k, 3) with integer coordinates in [0,7].
    """
    n = 8
    deadline = time.time() + 580  # safety margin: 630s limit, leave 50s

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # precompute |p|^2 for each grid point
    num_points = len(grid)

    def can_add(arr, p, arr_sq, p_sq):
        """Check if point p (int array) can be added to set arr without creating
        a co-spherical 5-tuple. arr_sq is precomputed |q|^2 for q in arr."""
        m = len(arr)
        if m < 4:
            return True
        # Build lifted vectors: (q - p, |q|^2 - |p|^2) for each q in arr
        # A 5-tuple {p, q1, q2, q3, q4} is co-spherical iff the 4x4 matrix
        # with rows (qi - p, |qi|^2 - |p|^2) for i=1..4 has det == 0.
        diff = arr - p  # (m, 3)
        sq_diff = arr_sq - p_sq  # (m,)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, :3] = diff
        w[:, 3] = sq_diff
        # Check all C(m,4) combinations
        if m < 4:
            return True
        # Optimization: If m is large, this is slow. But we must be correct.
        combos = list(itertools.combinations(range(m), 4))
        sub = w[np.array(combos)]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def greedy_pass(rng):
        """One greedy pass with random order. Returns (selected_points, skipped_indices)."""
        order = rng.permutation(num_points)
        selected = []
        selected_sq = []
        skipped = []
        for idx in order:
            p = grid[idx]
            p_sq = grid_sq[idx]
            # Convert lists to arrays for can_add
            arr = np.array(selected, dtype=np.int64) if selected else np.empty((0,3), dtype=np.int64)
            arr_sq = np.array(selected_sq, dtype=np.int64) if selected_sq else np.empty(0)
            
            if can_add(arr, p, arr_sq, p_sq):
                selected.append(p)
                selected_sq.append(p_sq)
            else:
                skipped.append(idx)
        
        arr = np.array(selected, dtype=np.int64) if selected else np.empty((0, 3), dtype=np.int64)
        return arr, skipped

    def local_improve(arr, skipped, grid_sq, rng, time_budget=2.0):
        """Try to improve arr by re-inserting skipped points or swaps."""
        if len(skipped) == 0:
            return arr
        arr_sq = (arr ** 2).sum(axis=1) if len(arr) > 0 else np.empty(0)
        improved = False
        start = time.time()
        
        # Try re-inserting each skipped point
        # Iterate over a copy to safely modify skipped
        for idx in list(skipped):
            if time.time() - start > time_budget:
                break
            if idx not in skipped: # Skip if already removed
                continue
            p = grid[idx]
            p_sq = grid_sq[idx]
            if can_add(arr, p, arr_sq, p_sq):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
                skipped.remove(idx)
                improved = True
        
        # Try simple swaps: remove one point, add two skipped
        if len(skipped) >= 2 and time.time() - start < time_budget:
            # Limit number of swap attempts
            swap_attempts = 0
            max_swaps = 50
            # We need to be careful with indices if we modify skipped
            # We will collect valid swaps and apply them
            # But the original code applied them immediately.
            # Let's stick to the original logic but fix the iteration.
            
            for i in range(len(arr)):
                if swap_attempts >= max_swaps or time.time() - start > time_budget:
                    break
                # Try removing arr[i]
                remaining = np.delete(arr, i, axis=0)
                remaining_sq = np.delete(arr_sq, i)
                
                # Try adding two skipped points
                # Use list(skipped) to iterate over a snapshot
                skipped_snapshot = list(skipped)
                found_swap = False
                for s1 in range(len(skipped_snapshot)):
                    if swap_attempts >= max_swaps or time.time() - start > time_budget or found_swap:
                        break
                    idx1 = skipped_snapshot[s1]
                    p1 = grid[idx1]
                    sq1 = grid_sq[idx1]
                    if not can_add(remaining, p1, remaining_sq, sq1):
                        continue
                    temp_arr = np.vstack([remaining, p1]) if len(remaining) else p1[None, :]
                    temp_sq = np.append(remaining_sq, sq1)
                    
                    for s2 in range(s1 + 1, len(skipped_snapshot)):
                        if swap_attempts >= max_swaps or time.time() - start > time_budget or found_swap:
                            break
                        idx2 = skipped_snapshot[s2]
                        p2 = grid[idx2]
                        sq2 = grid_sq[idx2]
                        if can_add(temp_arr, p2, temp_sq, sq2):
                            # Valid swap
                            new_arr = np.vstack([temp_arr, p2]) if len(temp_arr) else p2[None, :]
                            arr = new_arr
                            arr_sq = np.append(temp_sq, sq2)
                            # Remove from skipped
                            if idx1 in skipped:
                                skipped.remove(idx1)
                            if idx2 in skipped:
                                skipped.remove(idx2)
                            swap_attempts += 1
                            improved = True
                            found_swap = True
                            break
                        swap_attempts += 1
                    if not found_swap:
                        swap_attempts += 1
                if found_swap:
                    break
        
        return arr

    rng = np.random.default_rng(0)
    best = None
    best_len = 0
    
    num_restarts = 1000  # will be limited by deadline
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        arr, skipped = greedy_pass(rng)
        
        # Local improvement
        if time.time() < deadline - 10:
            arr = local_improve(arr, skipped, grid_sq, rng, time_budget=1.0)
        
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
            # print(f"Restart {r}: improved to {best_len} points", flush=True)
    
    if best is None:
        # Fallback: return empty or a small valid set
        return np.empty((0, 3), dtype=np.int64)
    
    return best.astype(np.int64)


# EVOLVE-BLOCK-END
