import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Find the maximum number of points in an 8x8x8 grid such that no 5 points 
    are co-spherical or co-planar.
    """
    n = 8
    deadline = time.monotonic() + 600.0
    safety_margin = 10.0
    
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    N = len(grid)
    
    # Map from point tuple to index
    point_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    all_indices = set(range(N))

    def check_validity(new_idx, current_indices):
        """
        Check if adding new_idx to current_indices (list of ints) is valid.
        """
        m = len(current_indices)
        if m < 4:
            return True
        
        p = grid[new_idx]
        p_sq = grid_sq[new_idx]
        
        # Extract points
        # To avoid creating a list of indices and then indexing grid, 
        # we can maintain the points array in the main loop?
        # No, current_indices changes.
        # Let's just index.
        pts = grid[np.array(current_indices, dtype=np.int64)]
        sqs = grid_sq[np.array(current_indices, dtype=np.int64)]
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = pts - p
        w[:, 3] = sqs - p_sq
        
        if m <= 60:
            combos = list(itertools.combinations(range(m), 4))
            chunk_size = 50000
            for i in range(0, len(combos), chunk_size):
                chunk = combos[i:i+chunk_size]
                if not chunk: continue
                indices = np.array(chunk, dtype=np.int64)
                sub_mats = w[indices]
                dets = np.linalg.det(sub_mats)
                if np.any(np.abs(dets) < 0.5):
                    return False
        else:
            # Sequential with early exit
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        return True

    def greedy_init(seed):
        rng = np.random.default_rng(seed)
        order = rng.permutation(N)
        current_indices = []
        for idx in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            if check_validity(idx, current_indices):
                current_indices.append(idx)
        return current_indices

    # Phase 1: Greedy Init
    best_indices = []
    for seed in range(10):
        if time.monotonic() + safety_margin >= deadline:
            break
        cand = greedy_init(seed)
        if len(cand) > len(best_indices):
            best_indices = cand

    if not best_indices:
        return np.empty((0, 3), dtype=np.int64)

    # Phase 2: Simulated Annealing
    current_indices = list(best_indices)
    current_set = set(current_indices)
    best_indices = list(current_indices)
    best_size = len(current_indices)
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while T > min_T and time.monotonic() + safety_margin < deadline:
        if len(current_set) < N:
            available = list(all_indices - current_set)
            if available:
                rand_idx = random.choice(available)
                
                if check_validity(rand_idx, current_indices):
                    current_indices.append(rand_idx)
                    current_set.add(rand_idx)
                    if len(current_indices) > best_size:
                        best_size = len(current_indices)
                        best_indices = list(current_indices)
                else:
                    # Kick
                    if random.random() < T and current_indices:
                        rem_idx = random.choice(current_indices)
                        current_indices.remove(rem_idx)
                        current_set.remove(rem_idx)
        
        T *= cooling_rate
        
    # Return best found
    final_indices = np.array(best_indices, dtype=np.int64)
    return grid[final_indices]

no_5_on_a_sphere8()
