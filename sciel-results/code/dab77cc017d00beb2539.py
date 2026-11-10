import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute: for each point p, and each 4-tuple (a,b,c,d) of other points,
    # store the determinant of the matrix with rows (q_i - p, |q_i|^2 - |p|^2)
    # This is expensive to precompute fully, but we can cache on demand.
    
    # Better: precompute the full N x N x 4 matrix W where W[p, q, :] = (grid[q] - grid[p], |q|^2 - |p|^2)
    # Then for a given p and 4 points a,b,c,d, we just need det of W[p, [a,b,c,d], :]
    
    # W shape: (N, N, 4)
    W = np.zeros((N, N, 4), dtype=np.float64)
    for p in range(N):
        W[p, :, 0:3] = grid - grid[p]
        W[p, :, 3] = sq_norms - sq_norms[p]
    
    def can_add_to_set(points_indices, new_point_idx):
        if len(points_indices) < 4:
            return True
        
        m = len(points_indices)
        p = new_point_idx
        
        # Get the rows for the 4 points
        # We need to check all C(m,4) subsets
        if m < 4:
            return True
        
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return True
        
        # For each combo, get the 4x4 matrix and compute det
        # Vectorize: sub has shape (C, 4, 4)
        idx_arr = np.array([points_indices[list(c)] for c in combos])  # (C, 4)
        sub = W[p, idx_arr, :]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def greedy_init(seed):
        np.random.seed(seed)
        order = np.random.permutation(N)
        selected_indices = []
        for idx in order:
            if can_add_to_set(selected_indices, idx):
                selected_indices.append(idx)
        return selected_indices

    def local_search(initial_indices, max_time):
        current_indices = list(initial_indices)
        current_set = set(current_indices)
        best_indices = list(current_indices)
        best_size = len(current_indices)
        
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            added = False
            candidates = [i for i in range(N) if i not in current_set]
            np.random.shuffle(candidates)
            
            for idx in candidates:
                if can_add_to_set(current_indices, idx):
                    current_indices.append(idx)
                    current_set.add(idx)
                    added = True
                    if len(current_indices) > best_size:
                        best_indices = list(current_indices)
                        best_size = len(current_indices)
                    break
            
            if added:
                continue
            
            swapped = False
            for _ in range(25):
                out_idx = np.random.choice(current_indices)
                remaining = [i for i in current_indices if i != out_idx]
                remaining_set = set(remaining)
                
                candidates = [i for i in range(N) if i not in remaining_set]
                np.random.shuffle(candidates)
                
                for in_idx in candidates[:30]:
                    if can_add_to_set(remaining, in_idx):
                        current_indices = remaining + [in_idx]
                        current_set = remaining_set | {in_idx}
                        swapped = True
                        break
                
                if swapped:
                    break
            
            if not swapped:
                two_swapped = False
                for _ in range(10):
                    out1 = np.random.choice(current_indices)
                    out2_candidates = [i for i in current_indices if i != out1]
                    out2 = np.random.choice(out2_candidates)
                    remaining = [i for i in current_indices if i not in {out1, out2}]
                    remaining_set = set(remaining)
                    
                    candidates = [i for i in range(N) if i not in remaining_set]
                    np.random.shuffle(candidates)
                    
                    for in1 in candidates[:20]:
                        temp = remaining + [in1]
                        for in2 in candidates[:20]:
                            if in2 == in1:
                                continue
                            if can_add_to_set(temp, in2):
                                current_indices = temp + [in2]
                                current_set = remaining_set | {in1, in2}
                                two_swapped = True
                                break
                        if two_swapped:
                            break
                    if two_swapped:
                        break
                
                if not two_swapped:
                    break

        return best_indices

    best_overall = []
    start_total = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    num_restarts = 0
    
    while time.time() - start_total < time_limit - safety_margin:
        num_restarts += 1
        seed = num_restarts
        init_indices = greedy_init(seed)
        
        elapsed = time.time() - start_total
        remaining = time_limit - safety_margin - elapsed
        if remaining < 10:
            break
        
        base_time = 300.0 / (2 ** (num_restarts - 1))
        ls_time = min(base_time, remaining)
        
        if ls_time < 1:
            break
        
        search_result = local_search(init_indices, ls_time)
        
        if len(search_result) > len(best_overall):
            best_overall = search_result
            
            current = list(best_overall)
            current_set = set(current)
            for idx in range(N):
                if idx not in current_set:
                    if can_add_to_set(current, idx):
                        current.append(idx)
                        current_set.add(idx)
            if len(current) > len(best_overall):
                best_overall = current

    if best_overall:
        current = list(best_overall)
        current_set = set(current)
        for idx in range(N):
            if idx not in current_set:
                if can_add_to_set(current, idx):
                    current.append(idx)
                    current_set.add(idx)
        best_overall = current

    if not best_overall:
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[best_overall].astype(np.int64)
