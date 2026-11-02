import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    # 1. Generate D11 Minimal Vectors
    # Type 1: Weight 1, +/- 2
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
            
    # Type 2: Weight 4, +/- 1, even sum (all 16 sign patterns valid for D11)
    vecs_t2 = []
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
            v = [0]*D
            for p, val in zip(pos, signs):
                v[p] = val
            vecs_t2.append(v)
                
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    V_t2 = np.array(vecs_t2, dtype=np.int64)
    N = len(V_t2)
    
    # 2. Compute Conflict Matrix for Type 2
    # Conflict if dot product > 2
    dots = V_t2 @ V_t2.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # 3. Find Independent Set using Deterministic Greedy
    # We use a simple greedy algorithm: pick vertex with lowest index (or lowest degree) that is not conflicted.
    # To make it "constructive" and deterministic, we process in index order.
    # However, min-degree greedy is often better. We'll do min-degree deterministic (tie-break by index).
    
    in_set = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    # Deterministic Greedy:
    # Repeatedly find the alive vertex with the minimum degree (tie-break: smallest index).
    # Add it to the set, remove it and its neighbors from alive.
    
    while np.any(alive):
        if time.time() - start_time > 110:
            break
            
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_idx])
        # Find indices with min degree
        cands = alive_idx[degrees[alive_idx] == min_deg]
        # Pick the one with smallest index (deterministic)
        chosen = cands[0]
        
        in_set[chosen] = True
        
        # Remove chosen and its neighbors from alive
        nbrs = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], nbrs)))
        alive[to_remove] = False
        
        # Update degrees of remaining alive vertices
        rem = np.where(alive)[0]
        if len(rem) > 0:
            # Subtract number of conflicts with removed vertices
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
    # Combine results
    V_t2_sel = V_t2[in_set]
    result = np.vstack([V_t1, V_t2_sel])
    
    return result
