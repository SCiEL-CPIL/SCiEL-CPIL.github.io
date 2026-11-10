import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590
    
    candidates = []
    norms = []
    
    # Norm 2
    for i in range(d):
        for j in range(i+1, d):
            for s_i in [1, -1]:
                for s_j in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s_i
                    v[j] = s_j
                    candidates.append(v)
                    norms.append(2)
                    
    # Norm 4
    for comb in combinations(range(d), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[comb[0]] = s1
                        v[comb[1]] = s2
                        v[comb[2]] = s3
                        v[comb[3]] = s4
                        candidates.append(v)
                        norms.append(4)
                        
    candidates = np.array(candidates)
    norms = np.array(norms)
    N = len(candidates)
    
    # Compute Dot Product Matrix
    dots = candidates @ candidates.T
    
    # Conflict Matrix
    # R_max = 4
    # (2,2): dot <= 0 -> conflict if dot > 0
    # (4,4): dot <= 2 -> conflict if dot > 2
    # (2,4): dot <= 1 -> conflict if dot > 1
    
    conflicts = np.zeros((N, N), dtype=bool)
    
    # (2,2)
    mask_2 = norms == 2
    mask_4 = norms == 4
    
    # We can vectorize this logic
    # Create a threshold matrix
    # If i is 2 and j is 2: thresh = 0
    # If i is 4 and j is 4: thresh = 2
    # If i is 2 and j is 4: thresh = 1
    # If i is 4 and j is 2: thresh = 1
    
    # norms_i = norms[:, None]
    # norms_j = norms[None, :]
    
    # thresh = np.where(norms_i == 2, 0, 2) # This is wrong for mixed
    # Let's just do it in blocks or with broadcasting
    
    # Simpler:
    # conflicts = dots > thresh
    
    thresh = np.zeros((N, N), dtype=np.int64)
    # (2,2) -> 0
    thresh[mask_2[:, None] & mask_2[None, :]] = 0
    # (4,4) -> 2
    thresh[mask_4[:, None] & mask_4[None, :]] = 2
    # (2,4) -> 1
    thresh[mask_2[:, None] & mask_4[None, :]] = 1
    # (4,2) -> 1
    thresh[mask_4[:, None] & mask_2[None, :]] = 1
    
    conflicts = dots > thresh
    np.fill_diagonal(conflicts, False)
    
    # Greedy + Local Search
    best_set = []
    
    for restart in range(20):
        if time.time() > deadline - 50:
            break
            
        order = np.random.permutation(N)
        current_selected = []
        current_mask = np.zeros(N, dtype=bool)
        conflict_counts = np.zeros(N, dtype=np.int32)
        
        for idx in order:
            if current_mask[idx]:
                continue
            if conflict_counts[idx] == 0:
                current_selected.append(idx)
                current_mask[idx] = True
                conflict_counts += conflicts[idx, :].astype(np.int32)
                
        improved = True
        while improved and time.time() < deadline - 50:
            improved = False
            for u_idx in current_selected[:]:
                current_mask[u_idx] = False
                conflict_counts -= conflicts[u_idx, :].astype(np.int32)
                
                candidates_to_swap = np.where((conflict_counts == 0) & (~current_mask))[0]
                
                if len(candidates_to_swap) > 0:
                    # Heuristic: pick one that allows more future additions?
                    # Or just random.
                    v_idx = candidates_to_swap[np.random.randint(len(candidates_to_swap))]
                    
                    current_selected.remove(u_idx)
                    current_selected.append(v_idx)
                    current_mask[v_idx] = True
                    conflict_counts += conflicts[v_idx, :].astype(np.int32)
                    improved = True
                    break
                    
        if len(current_selected) > len(best_set):
            best_set = current_selected[:]
            
    if not best_set:
        return np.array([np.zeros(d, dtype=np.int64) for _ in range(2)])
        
    points = candidates[best_set]
    return np.around(points).astype(np.int64)
