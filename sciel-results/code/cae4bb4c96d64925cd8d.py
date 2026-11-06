import numpy as np
import time
import random

N = 16
TOTAL = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 270  # 60s safety margin
    
    # Precompute coordinates
    pts = [(i // N, i % N) for i in range(TOTAL)]
    coords = np.array(pts, dtype=np.int32)
    
    # Precompute squared distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)  # 256x256
    
    # Precompute isosceles triple indicator
    d_ij = dist2[:, :, np.newaxis]
    d_ik = dist2[:, np.newaxis, :]
    d_jk = dist2[np.newaxis, :, :]
    is_iso = (d_ij == d_ik) | (d_ij == d_jk) | (d_ik == d_jk)
    del d_ij, d_ik, d_jk
    
    # Precompute identity mask
    eye = np.eye(TOTAL, dtype=bool)
    
    def can_add(p, S):
        if S.sum() < 2:
            return True
        mask = S[None, :] & S[:, None] & ~eye
        return not (is_iso[p] & mask).any()
    
    def check_maximal(S):
        S_sum = S.sum()
        if S_sum < 2:
            return S_sum == TOTAL
        for p in range(TOTAL):
            if S[p]:
                continue
            mask = S[None, :] & S[:, None] & ~eye
            if not (is_iso[p] & mask).any():
                return False
        return True
    
    def greedy_construct(order=None):
        if order is None:
            order = list(range(TOTAL))
        S = np.zeros(TOTAL, dtype=bool)
        for p in order:
            if can_add(p, S):
                S[p] = True
        return S
    
    # Initial greedy solution
    best_S = greedy_construct()
    best_size = int(best_S.sum())
    
    # Local search with restarts
    for restart in range(100):
        if time.time() > deadline:
            break
        
        order = list(range(TOTAL))
        random.shuffle(order)
        S = greedy_construct(order)
        size = int(S.sum())
        
        # Local search: repeatedly try to remove points
        while time.time() < deadline:
            indices = np.where(S)[0]
            random.shuffle(indices)
            improved = False
            for p in indices:
                if time.time() > deadline:
                    break
                S_new = S.copy()
                S_new[p] = False
                if check_maximal(S_new):
                    S = S_new
                    size -= 1
                    improved = True
            if not improved:
                break
        
        if size < best_size:
            best_size = size
            best_S = S.copy()
    
    return [pts[i] for i in np.where(best_S)[0]]
