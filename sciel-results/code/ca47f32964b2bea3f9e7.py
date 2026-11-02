import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    # 1. Generate Type 1 vectors (Weight 1, +/- 2)
    # These are always compatible with each other and with Type 2.
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
            
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    
    # 2. Generate Type 2 supports and signs
    # Supports are combinations of 4 indices from 11
    supports = list(itertools.combinations(range(D), 4))
    N_sup = len(supports) # 330
    
    # Signs for a support of size 4 with even sum
    # There are 8 such patterns.
    # We can generate them by iterating 16 masks and checking parity, or just listing them.
    # Masks 0..15. Sum of bits? No, sum of values (+1/-1).
    # Value sum is even if number of -1s is even.
    # Number of -1s is popcount(mask).
    # So we need popcount(mask) % 2 == 0.
    valid_masks = [m for m in range(16) if bin(m).count('1') % 2 == 0]
    
    # Precompute vectors for each support?
    # We need to select a subset of supports.
    
    # 3. Build Conflict Graph for Supports
    # Conflict if intersection size is 3.
    # Create list of sets for fast intersection
    supp_sets = [set(s) for s in supports]
    
    conflict_supp = np.zeros((N_sup, N_sup), dtype=bool)
    for i in range(N_sup):
        for j in range(i+1, N_sup):
            if len(supp_sets[i].intersection(supp_sets[j])) == 3:
                conflict_supp[i, j] = True
                conflict_supp[j, i] = True
    
    # 4. Find Max Independent Set in Support Graph
    # Use Greedy with Random Restarts
    best_supp_set = np.zeros(N_sup, dtype=bool)
    best_count = 0
    
    n_restarts = 1000
    for r in range(n_restarts):
        if time.time() - start_time > 110:
            break
            
        in_set = np.zeros(N_sup, dtype=bool)
        alive = np.ones(N_sup, dtype=bool)
        
        # Degrees
        degrees = conflict_supp.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            # Pick min degree, random tie-break
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = np.where(conflict_supp[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflict_supp[rem][:, to_remove].sum(axis=1)
                
        count = np.sum(in_set)
        if count > best_count:
            best_count = count
            best_supp_set = in_set.copy()
            
    # 5. Construct Final Vectors
    # Include all T1
    result_vecs = V_t1.tolist()
    
    # Include T2 for selected supports
    for i in range(N_sup):
        if best_supp_set[i]:
            supp = supports[i]
            for mask in valid_masks:
                signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
                v = [0]*D
                for p, val in zip(supp, signs):
                    v[p] = val
                result_vecs.append(v)
                
    result = np.array(result_vecs, dtype=np.int64)
    
    return result
