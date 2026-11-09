import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 580
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)
    num_points = len(grid)
    
    # Precompute all 4-tuples indices for a given m? No, m changes.
    
    def can_add(selected, selected_sq, p, p_sq):
        m = len(selected)
        if m < 4:
            return True
        
        # Convert to arrays if not already
        if not isinstance(selected, np.ndarray):
            sel_arr = np.array(selected, dtype=np.int64)
            sel_sq_arr = np.array(selected_sq, dtype=np.int64)
        else:
            sel_arr = selected
            sel_sq_arr = selected_sq
            
        diff = sel_arr - p
        sq_diff = sel_sq_arr - p_sq
        
        # Get combinations of indices
        # Optimization: If m is small, this is fast.
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return True
            
        idxs = np.array(combos)
        d = diff[idxs]
        s = sq_diff[idxs]
        
        M = np.empty((len(combos), 4, 4), dtype=np.float64)
        M[:, :, :3] = d
        M[:, :, 3] = s
        
        dets = np.linalg.det(M)
        return not np.any(np.abs(dets) < 0.5)

    def greedy_pass(order):
        selected = []
        selected_sq = []
        for idx in order:
            p = grid[idx]
            p_sq = grid_sq[idx]
            if can_add(selected, selected_sq, p, p_sq):
                selected.append(p)
                selected_sq.append(p_sq)
        
        return np.array(selected, dtype=np.int64) if selected else np.empty((0, 3), dtype=np.int64)

    rng = np.random.default_rng(42)
    best = None
    best_len = 0
    
    # Precompute center distances for heuristic ordering
    center = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center, axis=1)
    
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Strategy: Mix random and heuristic orders
        mode = r % 3
        if mode == 0:
            order = rng.permutation(num_points)
        elif mode == 1:
            # Far from center first
            order = np.argsort(dists)[::-1]
            order = order[rng.permutation(num_points)] # Shuffle within same distance? No, just permute
            # Actually, to keep the heuristic, we should sort by distance, then randomize ties?
            # Simpler: just use the sorted order, but randomize the whole thing slightly?
            # Let's just use the sorted order for mode 1.
            order = np.argsort(dists)[::-1]
        else:
            # Close to center first
            order = np.argsort(dists)
            
        arr = greedy_pass(order)
        
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
            
    if best is None:
        return np.empty((0, 3), dtype=np.int64)
        
    return best
