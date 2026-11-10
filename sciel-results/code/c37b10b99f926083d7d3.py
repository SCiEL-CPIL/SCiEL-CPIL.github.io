import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + 610
    rng = np.random.default_rng(42)
    
    def det_check(p, q1, q2, q3, q4):
        """Return |det| of the 4x4 matrix testing cosphericity/coplanarity."""
        mat = np.array([
            [q1[0]-p[0], q1[1]-p[1], q1[2]-p[2], grid_sq[q1.__hash__() if False else (q1[0]*64+q1[1]*8+q1[2])] - grid_sq[(p[0]*64+p[1]*8+p[2])]],
            [q2[0]-p[0], q2[1]-p[1], q2[2]-p[2], grid_sq[(q2[0]*64+q2[1]*8+q2[2])] - grid_sq[(p[0]*64+p[1]*8+p[2])]],
            [q3[0]-p[0], q3[1]-p[1], q3[2]-p[2], grid_sq[(q3[0]*64+q3[1]*8+q3[2])] - grid_sq[(p[0]*64+p[1]*8+p[2])]],
            [q4[0]-p[0], q4[1]-p[1], q4[2]-p[2], grid_sq[(q4[0]*64+q4[1]*8+q4[2])] - grid_sq[(p[0]*64+p[1]*8+p[2])]],
        ], dtype=np.float64)
        return abs(np.linalg.det(mat))
    
    def can_add_to_set(points_arr, p_idx):
        """Check if p_idx can be added to points_arr without creating a forbidden 5-tuple."""
        if len(points_arr) < 4:
            return True
        p = grid[p_idx]
        m = len(points_arr)
        # Check all 4-subsets of points_arr
        for combo in itertools.combinations(range(m), 4):
            q1_idx = points_arr[combo[0]]
            q2_idx = points_arr[combo[1]]
            q3_idx = points_arr[combo[2]]
            q4_idx = points_arr[combo[3]]
            d = det_check(p, grid[q1_idx], grid[q2_idx], grid[q3_idx], grid[q4_idx])
            if d < 0.5:
                return False
        return True
    
    def greedy_fill(order):
        """Greedily add points in the given order if valid."""
        result = []
        for idx in order:
            if can_add_to_set(np.array(result, dtype=np.int64) if result else np.array([], dtype=np.int64), idx):
                result.append(idx)
        return np.array(result, dtype=np.int64)
    
    def local_search_1opt(arr_idx, time_limit):
        """1-opt local search: try replacing each point with any other point."""
        current = list(arr_idx)
        current_set = set(current)
        best = current[:]
        best_len = len(best)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(current)):
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                
                # Try all candidates not in remaining
                for new_idx in range(N):
                    if new_idx in rem_set:
                        continue
                    
                    # Try adding new_idx to remaining
                    temp = remaining + [new_idx]
                    temp_arr = np.array(temp, dtype=np.int64)
                    if can_add_to_set(temp_arr[:-1] if len(temp_arr) > 0 else np.array([], dtype=np.int64), new_idx):
                        # Check full validity of temp
                        valid = True
                        for j in range(len(temp)):
                            if j == len(temp) - 1:
                                continue
                        # Actually, let's just verify the whole set
                        full_valid = True
                        for k in range(len(temp)):
                            p_k = grid[temp[k]]
                            others = [temp[m] for m in range(len(temp)) if m != k]
                            for combo in itertools.combinations(others, 4):
                                d = det_check(p_k, grid[combo[0]], grid[combo[1]], grid[combo[2]], grid[combo[3]])
                                if d < 0.5:
                                    full_valid = False
                                    break
                            if not full_valid:
                                break
                        if full_valid:
                            current = temp
                            current_set = set(current)
                            improved = True
                            if len(current) > best_len:
                                best = current[:]
                                best_len = len(best)
                            break
                if time.time() >= time_limit:
                    break
        
        return np.array(best, dtype=np.int64)
    
    best_arr = np.array([], dtype=np.int64)
    
    # Multiple restarts with different strategies
    center = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center, axis=1)
    sums = grid.sum(axis=1)
    
    while time.time() < deadline - 30:
        strategy = int(time.time() * 1000) % 6
        
        if strategy == 0:
            order = rng.permutation(N)
        elif strategy == 1:
            noise = rng.random(N)
            order = np.argsort(dists + noise)
        elif strategy == 2:
            noise = rng.random(N)
            order = np.argsort(-dists + noise)
        elif strategy == 3:
            noise = rng.random(N)
            order = np.argsort(sums + noise)
        elif strategy == 4:
            noise = rng.random(N)
            order = np.argsort(-sums + noise)
        else:
            # Random with spatial structure
            order = np.argsort(grid[:, 0] * 100 + grid[:, 1] * 10 + grid[:, 2] + rng.random(N) * 5)
        
        arr = greedy_fill(order)
        
        if len(arr) > 0 and time.time() < deadline - 40:
            ls_limit = min(deadline - 30, time.time() + 20)
            arr = local_search_1opt(arr, ls_limit)
        
        if len(arr) > len(best_arr):
            best_arr = arr.copy()
        
        # If we have a decent set, try more local search
        if len(best_arr) >= 25 and time.time() < deadline - 60:
            ls_limit = deadline - 35
            improved = local_search_1opt(best_arr, ls_limit)
            if len(improved) > len(best_arr):
                best_arr = improved.copy()
    
    return best_arr.astype(np.int64)
