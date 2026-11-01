import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    # 1. Generate D11 Minimal Vectors
    # Type 1: Weight 1, +/- 2
    # Type 2: Weight 4, +/- 1, even sum
    
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
            
    vecs_t2 = []
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            if bin(mask).count('1') % 2 == 0:
                signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
                v = [0]*D
                for p, val in zip(pos, signs):
                    v[p] = val
                vecs_t2.append(v)
                
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    V_t2 = np.array(vecs_t2, dtype=np.int64)
    N = len(V_t2)
    
    # 2. Compute Conflict Matrix for Type 2
    # Type 1 vectors are compatible with all, so we only need to solve for Type 2
    dots = V_t2 @ V_t2.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # 3. Find Independent Set using Randomized Greedy
    best_set = np.zeros(N, dtype=bool)
    best_count = 0
    
    # Run multiple restarts to find a better independent set
    n_restarts = 100
    for _ in range(n_restarts):
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            # Randomly pick among min-degree candidates
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        count = np.sum(in_set)
        if count > best_count:
            best_count = count
            best_set = in_set.copy()
            
    # Combine results
    V_t2_sel = V_t2[best_set]
    result = np.vstack([V_t1, V_t2_sel])
    
    return result
