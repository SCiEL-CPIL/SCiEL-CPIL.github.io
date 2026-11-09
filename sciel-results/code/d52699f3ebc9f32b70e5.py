import numpy as np
import time
import random

N = 16
GRID_SIZE = N * N
DEADLINE = time.time() + 280

points = [(x, y) for x in range(N) for y in range(N)]
D = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
for i in range(GRID_SIZE):
    x1, y1 = points[i]
    for j in range(GRID_SIZE):
        x2, y2 = points[j]
        D[i, j] = (x1 - x2) ** 2 + (y1 - y2) ** 2

def is_maximal(S_indices):
    S_set = set(S_indices)
    S_arr = np.array(S_indices)
    
    # Precompute distance sets for each point in S
    # dist_sets[i] is a set of distances from S[i] to other points in S
    dist_sets = []
    for idx in S_arr:
        # Get distances from idx to all points in S
        d = D[idx, S_arr]
        # Exclude 0 (distance to self)
        # d[0] is D[idx, idx] = 0 if idx is first? No, S_arr contains idx.
        # We want set of d[j] where S_arr[j] != idx.
        # Since D is symmetric, D[idx, idx] = 0.
        # So we can just take unique non-zero values.
        non_zero = d[d > 0]
        dist_sets.append(set(non_zero))
        
    for p in range(GRID_SIZE):
        if p in S_set:
            continue
            
        # Check Condition 1: Distinct distances
        d_p_s = D[p, S_arr]
        if len(d_p_s) > 1:
            # Check for duplicates
            # If any value appears > 1, blocked.
            # Using np.unique is fast enough for small arrays
            vals, counts = np.unique(d_p_s, return_counts=True)
            if np.any(counts > 1):
                continue
                
        # Check Condition 2: d(p, s) not in dist_sets[i] for all i
        blocked = False
        for i, s_idx in enumerate(S_arr):
            if d_p_s[i] in dist_sets[i]:
                blocked = True
                break
                
        if not blocked:
            return False
            
    return True

def construct():
    best_set = None
    best_size = GRID_SIZE + 1
    
    start_time = time.time()
    
    for restart in range(2000):
        if time.time() > DEADLINE:
            break
            
        order = list(range(GRID_SIZE))
        random.shuffle(order)
        
        S = []
        for p in order:
            can_add = True
            if len(S) >= 2:
                S_arr = np.array(S)
                d_p_s = D[p, S_arr]
                
                # Check duplicates
                if len(d_p_s) > 1:
                    vals, counts = np.unique(d_p_s, return_counts=True)
                    if np.any(counts > 1):
                        can_add = False
                
                if can_add:
                    # Precompute dist sets for current S?
                    # This is expensive to do every time.
                    # But S is small.
                    # Or just do the O(S^2) check.
                    for i, v in enumerate(S):
                        dv_p = D[p, v]
                        # Check if dv_p is in distances from v to other points in S
                        # We can just loop u in S
                        for u in S:
                            if u == v: continue
                            if D[v, u] == dv_p:
                                can_add = False
                                break
                        if not can_add: break
            
            if can_add:
                S.append(p)
                
        changed = True
        while changed:
            if time.time() > DEADLINE:
                if best_set is None or len(S) < len(best_set):
                    best_set = S[:]
                return best_set
            changed = False
            for i in range(len(S)):
                if time.time() > DEADLINE:
                    if best_set is None or len(S) < len(best_set):
                        best_set = S[:]
                    return best_set
                S_prime = S[:i] + S[i+1:]
                if is_maximal(S_prime):
                    S = S_prime
                    changed = True
                    break
                    
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    if best_set is None:
        return S
    return best_set
