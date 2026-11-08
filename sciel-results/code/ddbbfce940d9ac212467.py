import time
import numpy as np
import itertools
import random

D = 11

def construct():
    deadline = time.time() + 110
    
    candidates = []
    seen = set()
    
    def add_candidate(v):
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            candidates.append(v)
            
    # Generate Norm^2 <= 5
    # 1 nonzero: (1,0,...)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            add_candidate(v)
            
    # 2 nonzero: (1,1,0...), (2,1,0...), (1,2,0...)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                add_candidate(v)
        for s2 in (1, -1):
            for s1 in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * s2
                v[j] = s1
                add_candidate(v)
                v2 = np.zeros(D, dtype=np.int64)
                v2[i] = s1
                v2[j] = 2 * s2
                add_candidate(v2)
                
    # 3 nonzero: (1,1,1,0...)
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    add_candidate(v)

    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Precompute distance matrix
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    dist_sq = np.rint(dist_sq).astype(np.int64)
    
    best_result = None
    best_k = 0
    
    # Possible max norms squared in this pool: 1, 2, 5
    for R in [1, 2, 5]:
        if time.time() > deadline - 10:
            break
            
        mask = norms_sq <= R
        cand_idx = np.where(mask)[0]
        k_sub = len(cand_idx)
        
        if k_sub == 0:
            continue
        
        best_for_R = []
        best_k_for_R = 0
        
        # Run Iterated Greedy
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            
            # Greedy Init with random order
            selected = []
            is_selected = np.zeros(n_cand, dtype=bool)
            
            order = np.random.permutation(k_sub)
            
            for idx_pos in order:
                v_idx = cand_idx[idx_pos]
                if is_selected[v_idx]:
                    continue
                
                if not selected:
                    selected.append(v_idx)
                    is_selected[v_idx] = True
                    continue
                    
                # Check compatibility
                # dist_sq[v_idx, selected] >= R
                dists = dist_sq[v_idx, selected]
                if np.min(dists) >= R:
                    selected.append(v_idx)
                    is_selected[v_idx] = True
                    
            k_sel = len(selected)
            if k_sel > best_k_for_R:
                best_k_for_R = k_sel
                best_for_R = selected[:]
                
            # Local Search: Try to add more
            # Find all unselected candidates
            unselected = [idx for idx in cand_idx if not is_selected[idx]]
            # Shuffle unselected
            random.shuffle(unselected)
            
            for v_idx in unselected:
                if time.time() > deadline - 10:
                    break
                dists = dist_sq[v_idx, selected]
                if np.min(dists) >= R:
                    selected.append(v_idx)
                    is_selected[v_idx] = True
                    
            k_sel = len(selected)
            if k_sel > best_k_for_R:
                best_k_for_R = k_sel
                best_for_R = selected[:]
                
        if best_for_R and best_k_for_R > best_k:
            best_k = best_k_for_R
            best_result = candidates[best_for_R]
            
    if best_result is None:
        # Fallback: return empty or small valid set
        # Norm 1 vectors are always valid for R=1
        best_result = candidates[:22]
        
    return best_result
