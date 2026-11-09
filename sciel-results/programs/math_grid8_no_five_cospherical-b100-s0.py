# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from math import comb

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    def can_add_fast(arr: np.ndarray, p_idx: int, arr_sq: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        diff = arr - p  # (m, 3)
        sq_diff = arr_sq - p_sq  # (m,)
        
        # Generate all 4-combinations
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return True
            
        sub_diff = diff[combos]  # (C, 4, 3)
        sub_sq = sq_diff[combos]  # (C, 4)
        mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)  # (C, 4, 4)
        dets = np.linalg.det(mat)  # (C,)
        
        return not np.any(np.abs(dets) < 0.5)

    def greedy_fill(order_idx: np.ndarray) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.int64)
        
        for idx in order_idx:
            if can_add_fast(arr, idx, arr_sq):
                p = grid[idx]
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                arr_sq = np.append(arr_sq, grid_sq[idx])
                
        return arr

    def local_search(arr: np.ndarray, max_iter: int = 30, time_limit: float = None) -> np.ndarray:
        arr_sq = (arr ** 2).sum(axis=1)
        arr_set = set(tuple(p) for p in arr)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            iterations += 1
            for i in range(len(arr)):
                # Try removing arr[i] and adding a new point
                remaining = np.delete(arr, i, axis=0)
                remaining_sq = np.delete(arr_sq, i)
                removed_point = tuple(arr[i])
                
                # Try adding each grid point not in remaining
                for new_idx in range(N):
                    if tuple(grid[new_idx]) in arr_set and tuple(grid[new_idx]) != removed_point:
                        continue
                    if can_add_fast(remaining, new_idx, remaining_sq):
                        new_arr = np.vstack([remaining, grid[new_idx]])
                        return new_arr
            # If no single swap found, try 2-swap: remove 2 points, add 2 new points
            if len(arr) >= 2:
                for i in range(len(arr)):
                    for j in range(i + 1, len(arr)):
                        remaining = np.delete(arr, [i, j], axis=0)
                        remaining_sq = np.delete(arr_sq, [i, j])
                        removed1 = tuple(arr[i])
                        removed2 = tuple(arr[j])
                        
                        # Try adding pairs of new points
                        for new_idx1 in range(N):
                            p1 = tuple(grid[new_idx1])
                            if p1 in arr_set and p1 not in (removed1, removed2):
                                continue
                            rem1 = np.vstack([remaining, grid[new_idx1]])
                            rem1_sq = np.append(remaining_sq, grid_sq[new_idx1])
                            if not can_add_fast(rem1, new_idx1, rem1_sq):
                                continue
                            for new_idx2 in range(N):
                                p2 = tuple(grid[new_idx2])
                                if p2 in arr_set and p2 not in (removed1, removed2):
                                    continue
                                if p2 == p1:
                                    continue
                                if can_add_fast(rem1, new_idx2, rem1_sq):
                                    new_arr = np.vstack([rem1, grid[new_idx2]])
                                    return new_arr
        return arr

    best_arr = np.empty((0, 3), dtype=np.int64)
    start_time = time.time()
    time_limit = 630
    safety_margin = 20
    deadline = start_time + time_limit - safety_margin
    
    rng = np.random.default_rng(42)
    
    center = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center, axis=1)
    sums = grid.sum(axis=1)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        strategy = restart_count % 9
        
        if strategy == 0:
            order_idx = rng.permutation(N)
        elif strategy == 1:
            noise = rng.random(N)
            order_idx = np.argsort(dists + noise)
        elif strategy == 2:
            noise = rng.random(N)
            order_idx = np.argsort(sums + noise)
        elif strategy == 3:
            noise = rng.random(N)
            order_idx = np.argsort(-sums + noise)
        elif strategy == 4:
            noise = rng.random(N)
            order_idx = np.argsort(-dists + noise)
        elif strategy == 5:
            order_idx = rng.permutation(N)
        elif strategy == 6:
            key = (grid[:, 0] * 13 + grid[:, 1] * 7 + grid[:, 2] * 11) % 100
            noise = rng.random(N)
            order_idx = np.argsort(key + noise)
        elif strategy == 7:
            prods = grid[:, 0] * grid[:, 1] * grid[:, 2]
            noise = rng.random(N)
            order_idx = np.argsort(prods + noise)
        else:
            maxes = grid.max(axis=1)
            noise = rng.random(N)
            order_idx = np.argsort(maxes + noise)
            
        current_arr = greedy_fill(order_idx)
        
        # Try local search if we have time
        if time.time() < deadline - 40 and len(current_arr) >= 20:
            ls_time_limit = deadline - 25
            current_arr = local_search(current_arr, max_iter=15, time_limit=ls_time_limit)
        
        if len(current_arr) > len(best_arr):
            best_arr = current_arr.copy()
            
        # If we've found a good solution, try more local search
        if len(best_arr) >= 30 and time.time() < deadline - 60:
            ls_time_limit = deadline - 25
            improved = local_search(best_arr, max_iter=10, time_limit=ls_time_limit)
            if len(improved) > len(best_arr):
                best_arr = improved.copy()
    
    return best_arr.astype(np.int64)

# EVOLVE-BLOCK-END
