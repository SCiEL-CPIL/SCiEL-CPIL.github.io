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
        
        # Generate all 4-combinations with chunking for memory safety
        chunk_size = 50000
        total_combos = comb(m, 4)
        
        for start in range(0, total_combos, chunk_size):
            end = min(start + chunk_size, total_combos)
            combos = list(itertools.islice(itertools.combinations(range(m), 4), start, end))
            if not combos:
                break
            combos_arr = np.array(combos, dtype=np.int64)
            sub_diff = diff[combos_arr]  # (C, 4, 3)
            sub_sq = sq_diff[combos_arr]  # (C, 4)
            mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2)  # (C, 4, 4)
            dets = np.linalg.det(mat)  # (C,)
            if np.any(np.abs(dets) < 0.5):
                return False
        
        return True

    def greedy_fill(order_idx: np.ndarray) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.int64)
        arr_indices = []
        
        for idx in order_idx:
            if can_add_fast(arr, idx, arr_sq):
                p = grid[idx]
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                arr_sq = np.append(arr_sq, grid_sq[idx])
                arr_indices.append(idx)
                
        return arr

    # Improved local search: try 1-for-1 swaps, then 2-for-1 swaps
    def local_search(arr: np.ndarray, max_iter: int = 30, time_limit: float = 600.0) -> np.ndarray:
        start = time.time()
        arr_sq = (arr ** 2).sum(axis=1)
        
        # Build index lookup once
        point_to_idx = {}
        for idx in range(N):
            key = (int(grid[idx][0]), int(grid[idx][1]), int(grid[idx][2]))
            point_to_idx[key] = idx
        
        arr_indices = set()
        for p in arr:
            key = (int(p[0]), int(p[1]), int(p[2]))
            arr_indices.add(point_to_idx[key])
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < time_limit:
            improved = False
            iterations += 1
            
            # 1-for-1 swap
            for i in range(len(arr)):
                key_i = (int(arr[i][0]), int(arr[i][1]), int(arr[i][2]))
                removed_idx = point_to_idx[key_i]
                remaining = np.delete(arr, i, axis=0)
                remaining_sq = np.delete(arr_sq, i)
                remaining_indices = arr_indices - {removed_idx}
                
                for new_idx in range(N):
                    if new_idx in remaining_indices:
                        continue
                    if can_add_fast(remaining, new_idx, remaining_sq):
                        new_arr = np.vstack([remaining, grid[new_idx]])
                        new_arr_sq = (new_arr ** 2).sum(axis=1)
                        new_arr_indices = remaining_indices | {new_idx}
                        arr = new_arr
                        arr_sq = new_arr_sq
                        arr_indices = new_arr_indices
                        improved = True
                        break
                if improved:
                    break
            
            # If no 1-for-1 swap, try 2-for-1 swap (remove 2, add 1)
            if not improved and len(arr) >= 5 and time.time() < time_limit - 10:
                for i in range(len(arr)):
                    for j in range(i+1, len(arr)):
                        key_i = (int(arr[i][0]), int(arr[i][1]), int(arr[i][2]))
                        key_j = (int(arr[j][0]), int(arr[j][1]), int(arr[j][2]))
                        rem_i = point_to_idx[key_i]
                        rem_j = point_to_idx[key_j]
                        remaining = np.delete(arr, [i, j], axis=0)
                        remaining_sq = np.delete(arr_sq, [i, j])
                        remaining_indices = arr_indices - {rem_i, rem_j}
                        
                        for new_idx in range(N):
                            if new_idx in remaining_indices:
                                continue
                            if can_add_fast(remaining, new_idx, remaining_sq):
                                new_arr = np.vstack([remaining, grid[new_idx]])
                                new_arr_sq = (new_arr ** 2).sum(axis=1)
                                new_arr_indices = remaining_indices | {new_idx}
                                arr = new_arr
                                arr_sq = new_arr_sq
                                arr_indices = new_arr_indices
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return arr

    # Re-insertion pass: try to add back points that were skipped
    def reinsert_pass(arr: np.ndarray, order_idx: np.ndarray, time_limit: float = 600.0) -> np.ndarray:
        start = time.time()
        arr_sq = (arr ** 2).sum(axis=1)
        
        # Build index lookup once
        point_to_idx = {}
        for idx in range(N):
            key = (int(grid[idx][0]), int(grid[idx][1]), int(grid[idx][2]))
            point_to_idx[key] = idx
        
        arr_set = set()
        for p in arr:
            key = (int(p[0]), int(p[1]), int(p[2]))
            arr_set.add(point_to_idx[key])
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for idx in order_idx:
                if idx in arr_set:
                    continue
                if can_add_fast(arr, idx, arr_sq):
                    arr = np.vstack([arr, grid[idx]])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
                    arr_set.add(idx)
                    changed = True
        
        return arr

    best_arr = np.empty((0, 3), dtype=np.int64)
    start_time = time.time()
    time_limit = 630
    safety_margin = 15
    deadline = start_time + time_limit - safety_margin
    
    rng = np.random.default_rng(42)
    
    center = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center, axis=1)
    sums = grid.sum(axis=1)
    
    # Precompute a good base ordering (distance from center, then sum)
    base_order = np.argsort(dists + 0.01 * sums)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        strategy = restart_count % 7
        
        if strategy == 0:
            order_idx = rng.permutation(N)
        elif strategy == 1:
            noise = rng.random(N) * 2.0
            order_idx = np.argsort(dists + noise)
        elif strategy == 2:
            noise = rng.random(N) * 2.0
            order_idx = np.argsort(sums + noise)
        elif strategy == 3:
            noise = rng.random(N) * 2.0
            order_idx = np.argsort(-sums + noise)
        elif strategy == 4:
            noise = rng.random(N) * 2.0
            order_idx = np.argsort(-dists + noise)
        elif strategy == 5:
            # Use base order with slight perturbation
            noise = rng.random(N) * 0.5
            order_idx = np.argsort(dists + 0.01 * sums + noise)
        else:
            key = (grid[:, 0] * 13 + grid[:, 1] * 7 + grid[:, 2] * 11) % 100
            noise = rng.random(N)
            order_idx = np.argsort(key + noise)
            
        current_arr = greedy_fill(order_idx)
        
        # Re-insertion pass
        if time.time() < deadline - 20:
            current_arr = reinsert_pass(current_arr, order_idx, deadline - 10)
        
        # Local search if we have time
        if time.time() < deadline - 30 and len(current_arr) >= 35:
            current_arr = local_search(current_arr, max_iter=25, time_limit=deadline - 10)
        
        if len(current_arr) > len(best_arr):
            best_arr = current_arr
            # print(f"Restart {restart_count}: found {len(best_arr)} points")
    
    # Final reinsertion on best_arr if time permits
    if time.time() < deadline - 5 and len(best_arr) > 0:
        final_order = np.argsort(dists)
        best_arr = reinsert_pass(best_arr, final_order, deadline - 2)
    
    return best_arr.astype(np.int64)

# EVOLVE-BLOCK-END
