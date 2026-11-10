# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    8x8x8 integer grid such that no five of the selected points are co-spherical
    (or co-planar).

    Uses randomized greedy search with multiple restarts, batch determinant
    checks, pruning, and a local improvement phase with simulated annealing
    using conflict-guided swaps.
    """
    n = 8
    deadline = time.time() + 615  # Safety margin before 630s limit

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    sq_norms_grid = (grid ** 2).sum(axis=1)

    def can_add(arr: np.ndarray, p: np.ndarray, sq_p: float = None) -> bool:
        m = len(arr)
        if m < 4:
            return True
        if sq_p is None:
            sq_p = (p ** 2).sum()
        
        diff = arr - p
        sq_diff = (arr ** 2).sum(axis=1) - sq_p
        W = np.column_stack((diff, sq_diff))
        
        if m == 4:
            det = np.linalg.det(W)
            return abs(det) >= 0.5
        
        combos = list(itertools.combinations(range(m), 4))
        sub = W[combos]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def find_conflicts(arr: np.ndarray, p: np.ndarray, sq_p: float = None, threshold: float = 2.0) -> list:
        """Find 4-subsets of arr that are close to being cospherical with p."""
        m = len(arr)
        if m < 4:
            return []
        if sq_p is None:
            sq_p = (p ** 2).sum()
        
        diff = arr - p
        sq_diff = (arr ** 2).sum(axis=1) - sq_p
        W = np.column_stack((diff, sq_diff))
        
        if m == 4:
            det = np.linalg.det(W)
            if abs(det) < threshold:
                return [tuple(range(4))]
            return []
        
        combos = list(itertools.combinations(range(m), 4))
        sub = W[combos]
        dets = np.linalg.det(sub)
        conflicts = []
        for i, det in enumerate(dets):
            if abs(det) < threshold:
                conflicts.append(combos[i])
        return conflicts

    def get_point_index(arr: np.ndarray, p: np.ndarray) -> int:
        """Find index of point p in arr, or -1 if not found."""
        for i in range(len(arr)):
            if np.array_equal(arr[i], p):
                return i
        return -1

    def local_improve_sa(arr: np.ndarray, rng: np.random.Generator, time_limit: float) -> np.ndarray:
        """
        Simulated annealing with conflict-guided swaps.
        At each step, identify points involved in near-conflicts, try removing
        one and adding a different point. Accept if valid and (improving or 
        probabilistically accepted based on temperature).
        """
        best_arr = arr.copy()
        best_len = len(best_arr)
        current_arr = arr.copy()
        current_len = len(current_arr)
        
        # Track which grid indices are in current set
        in_set = np.zeros(num_points, dtype=bool)
        for i in range(current_len):
            idx = get_point_index(grid, current_arr[i])
            if idx >= 0:
                in_set[idx] = True
        
        temperature = 1.0
        cooling_rate = 0.995
        
        while time.time() < time_limit and temperature > 0.01:
            # Find points with conflicts
            conflicted_points = set()
            for i in range(current_len):
                p = current_arr[i]
                others = np.delete(current_arr, i, axis=0)
                sq_p = (p ** 2).sum()
                # Check if adding p to others would create a conflict
                # i.e., find 4-subsets of others close to cospherical with p
                if len(others) >= 4:
                    conflicts = find_conflicts(others, p, sq_p, threshold=2.0)
                    if conflicts:
                        conflicted_points.add(i)
            
            if not conflicted_points:
                # No conflicts, try random swaps to see if we can improve
                if current_len < best_len:
                    temperature *= cooling_rate
                    continue
            
            # Pick a random conflicted point to remove
            remove_idx = rng.choice(list(conflicted_points))
            removed_p = current_arr[remove_idx]
            
            # Remove it
            current_arr = np.delete(current_arr, remove_idx, axis=0)
            current_len -= 1
            removed_grid_idx = get_point_index(grid, removed_p)
            if removed_grid_idx >= 0:
                in_set[removed_grid_idx] = False
            
            # Try to add a new point
            candidates = np.where(~in_set)[0]
            if len(candidates) == 0:
                # Can't add anything, revert
                current_arr = np.vstack([current_arr, removed_p]) if current_len > 0 else removed_p[None, :]
                current_len += 1
                if removed_grid_idx >= 0:
                    in_set[removed_grid_idx] = True
                temperature *= cooling_rate
                continue
            
            # Shuffle candidates and try to add one
            rng.shuffle(candidates)
            added = False
            for j in candidates[:50]:  # Try up to 50 candidates
                p = grid[j]
                if can_add(current_arr, p, sq_norms_grid[j]):
                    current_arr = np.vstack([current_arr, p]) if current_len > 0 else p[None, :]
                    current_len += 1
                    in_set[j] = True
                    added = True
                    break
            
            if not added:
                # Revert
                current_arr = np.vstack([current_arr, removed_p]) if current_len > 0 else removed_p[None, :]
                current_len += 1
                if removed_grid_idx >= 0:
                    in_set[removed_grid_idx] = True
                temperature *= cooling_rate
                continue
            
            # Accept if improving, or probabilistically if not
            if current_len > best_len:
                best_arr = current_arr.copy()
                best_len = current_len
                temperature *= 0.9  # Cool faster on improvement
            elif current_len < best_len:
                # Accept with probability based on temperature
                if rng.random() > np.exp(-(best_len - current_len) / temperature):
                    # Revert
                    current_arr = best_arr.copy()
                    current_len = best_len
                    in_set = np.zeros(num_points, dtype=bool)
                    for i in range(current_len):
                        idx = get_point_index(grid, current_arr[i])
                        if idx >= 0:
                            in_set[idx] = True
            else:
                # Same length, accept with some probability
                if rng.random() < 0.5:
                    pass  # Keep current
                else:
                    # Revert
                    current_arr = np.vstack([np.delete(current_arr, current_len - 1, axis=0), removed_p]) if current_len > 1 else removed_p[None, :]
                    current_len = best_len
                    in_set = np.zeros(num_points, dtype=bool)
                    for i in range(current_len):
                        idx = get_point_index(grid, current_arr[i])
                        if idx >= 0:
                            in_set[idx] = True
            
            temperature *= cooling_rate
        
        # Return best found
        return best_arr

    def local_improve_simple(arr: np.ndarray, rng: np.random.Generator, time_limit: float) -> np.ndarray:
        """Try to improve the set by removing one point and adding two others."""
        best_arr = arr.copy()
        best_len = len(best_arr)
        
        while time.time() < time_limit:
            improved = False
            m = len(best_arr)
            if m < 3:
                break
            
            # Try removing each point and see if we can add 2 more
            for i in range(m):
                if time.time() >= time_limit:
                    break
                trial = np.delete(best_arr, i, axis=0)
                removed_p = best_arr[i]
                
                # Try to add points from the remaining grid
                candidates = [j for j in range(num_points) if not np.any((grid[j] == removed_p).all())]
                rng.shuffle(candidates)
                
                added = 0
                for j in candidates:
                    if time.time() >= time_limit:
                        break
                    if added >= 2:
                        break
                    p = grid[j]
                    if can_add(trial, p, sq_norms_grid[j]):
                        trial = np.vstack([trial, p]) if len(trial) > 0 else p[None, :]
                        added += 1
                
                if len(trial) > best_len:
                    best_arr = trial
                    best_len = len(trial)
                    improved = True
            
            if not improved:
                break
        
        return best_arr

    rng = np.random.default_rng(42)
    best = None
    best_len = 0

    restart = 0
    while time.time() < deadline - 30:
        restart += 1
        
        # Random greedy pass
        order = rng.permutation(num_points)
        arr = np.empty((0, 3), dtype=np.int64)
        
        for i, idx in enumerate(order):
            # Pruning: if current size + remaining <= best size, break
            if len(arr) + (num_points - i) <= best_len:
                break
            
            p = grid[idx]
            if can_add(arr, p, sq_norms_grid[idx]):
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
        
        # Local improvement phases
        remaining_time = deadline - 30 - time.time()
        if remaining_time > 5 and len(arr) >= best_len:
            # First do simple local improvement
            simple_time = min(remaining_time * 0.2, 20)
            arr = local_improve_simple(arr, rng, time.time() + simple_time)
            
            # Then do simulated annealing with conflict guidance
            remaining_time = deadline - 30 - time.time()
            if remaining_time > 5:
                sa_time = min(remaining_time * 0.5, 60)
                arr = local_improve_sa(arr, rng, time.time() + sa_time)
        
        if best is None or len(arr) > best_len:
            best = arr.copy()
            best_len = len(arr)
        
        # If we've found a very large set, we can be more conservative
        if best_len >= 40 and time.time() > deadline - 120:
            break

    return best.astype(np.int64) if best is not None else np.empty((0, 3), dtype=np.int64)


# EVOLVE-BLOCK-END
