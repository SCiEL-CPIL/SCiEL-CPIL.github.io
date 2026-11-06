import numpy as np
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # 1. Precompute grid points and pairwise squared distances
    points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    num_points = len(points)
    
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    D_grid = np.sum(diffs ** 2, axis=2).astype(np.int32)
    
    # 2. Batch Validity Check
    def is_valid_batch(candidate_indices, S_indices):
        """
        candidate_indices: array of shape (C,)
        S_indices: array of shape (k,)
        Returns: boolean array of shape (C,)
        """
        C = len(candidate_indices)
        k = len(S_indices)
        
        if k < 2:
            return np.ones(C, dtype=bool)
        
        # d_CS: (C, k)
        d_CS = D_grid[np.ix_(candidate_indices, S_indices)]
        # d_SS: (k, k)
        d_SS = D_grid[np.ix_(S_indices, S_indices)]
        
        mask = ~np.eye(k, dtype=bool)
        
        # Cond 1: d(C, S_i) == d(C, S_j)
        # d_CS[:, :, None] (C, k, 1) vs d_CS[:, None, :] (C, 1, k) -> (C, k, k)
        cond1 = np.any((d_CS[:, :, np.newaxis] == d_CS[:, np.newaxis, :]) & mask[np.newaxis, :, :], axis=(1, 2))
        
        # Cond 2: d(C, S_i) == d(S_i, S_j)
        # d_CS[:, :, None] (C, k, 1) vs d_SS[None, :, :] (1, k, k) -> (C, k, k)
        cond2 = np.any((d_CS[:, :, np.newaxis] == d_SS[np.newaxis, :, :]) & mask[np.newaxis, :, :], axis=(1, 2))
        
        # Cond 3: d(C, S_j) == d(S_i, S_j)
        # d_CS[:, None, :] (C, 1, k) vs d_SS[None, :, :] (1, k, k) -> (C, k, k)
        cond3 = np.any((d_CS[:, np.newaxis, :] == d_SS[np.newaxis, :, :]) & mask[np.newaxis, :, :], axis=(1, 2))
        
        return ~(cond1 | cond2 | cond3)

    best_set = []
    best_len = 0
    
    # Baseline
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    baseline_indices = np.array([x for x in xs])
    best_set = baseline_indices.tolist()
    best_len = len(best_set)
    
    num_restarts = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_set = []
        current_indices = np.empty(0, dtype=np.int32)
        remaining = set(range(num_points))
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Sample candidates
            C = min(100, len(remaining))
            candidates = np.random.choice(list(remaining), size=C, replace=False)
            
            valid_mask = is_valid_batch(candidates, current_indices)
            
            if np.any(valid_mask):
                # Pick a valid candidate
                # Heuristic: pick the one that is "best"
                # For now, just pick the first one
                valid_candidates = candidates[valid_mask]
                # To add some diversity, pick randomly from valid
                chosen = np.random.choice(valid_candidates)
                
                current_set.append(chosen)
                current_indices = np.append(current_indices, chosen)
                remaining.remove(chosen)
            else:
                # No valid candidates in this sample
                # Try a smaller sample or break?
                # If C is small, maybe break.
                if C < 10:
                    break
                # Otherwise, try again with a new sample
                # But to avoid infinite loop, we can just break or reduce C
                C = max(10, C // 2)
                # Actually, if 100 are invalid, likely many more are.
                # Let's just break to save time for next restart.
                break
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
    result = []
    for idx in best_set:
        y = idx // N
        x = idx % N
        result.append((x, y))
        
    return result
# EVOLVE-BLOCK-END
