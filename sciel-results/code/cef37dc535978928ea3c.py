import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N
MAX_TIME = 280.0
MAX_DIST_SQ = 2 * (N - 1) ** 2
NUM_WORDS = (MAX_DIST_SQ + 64) // 64

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    xs = np.arange(N)
    ys = np.arange(N)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    all_indices = np.arange(NUM_POINTS)
    
    def check_and_add(candidate, S_points, M, S_size):
        """
        Check if candidate can be added to S.
        If yes, return updated S_points, M, S_size.
        If no, return original.
        """
        if S_size == 0:
            S_points_new = np.array([[candidate[0], candidate[1]]], dtype=np.int32)
            M_new = np.zeros((1, NUM_WORDS), dtype=np.uint64)
            return S_points_new, M_new, 1, True
        
        # Compute distances
        dx = candidate[0] - S_points[:, 0]
        dy = candidate[1] - S_points[:, 1]
        d2 = dx**2 + dy**2
        
        # Check uniqueness
        if len(np.unique(d2)) != S_size:
            return S_points, M, S_size, False
        
        # Check conflicts
        words = d2 // 64
        bits = 1 << (d2 % 64)
        
        # M is (K, W). We want M[i, words[i]] & bits[i]
        M_selected = M[np.arange(S_size), words]
        conflicts = M_selected & bits
        
        if np.any(conflicts):
            return S_points, M, S_size, False
        
        # Valid
        S_points_new = np.vstack([S_points, candidate])
        
        M_new = M.copy()
        M_new[np.arange(S_size), words] |= bits
        
        M_row = np.zeros(NUM_WORDS, dtype=np.uint64)
        M_row[words] |= bits
        M_new = np.vstack([M_new, M_row])
        
        return S_points_new, M_new, S_size + 1, True

    best_size = 0
    best_S_points = np.empty((0, 2), dtype=np.int32)
    
    # Greedy with random restarts
    while time.time() < deadline - 5:
        random.shuffle(all_indices)
        
        S_points = np.empty((0, 2), dtype=np.int32)
        M = np.zeros((0, NUM_WORDS), dtype=np.uint64)
        S_size = 0
        
        for idx in all_indices:
            if time.time() > deadline - 2:
                break
            candidate = coords[idx]
            S_points, M, S_size, added = check_and_add(candidate, S_points, M, S_size)
            if not added and S_size > 0:
                # Optional: break early if we've tried many and added few?
                # No, just keep going.
                pass
        
        if S_size > best_size:
            best_size = S_size
            best_S_points = S_points.copy()
            
    return [tuple(p) for p in best_S_points]
