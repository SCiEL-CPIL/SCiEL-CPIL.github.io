"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # 1. Precompute all 4-element subsets of [1, 16] as bitmasks
    all_sets_masks = []
    for subset in combinations(range(1, 17), 4):
        mask = 0
        for x in subset:
            mask |= (1 << (x - 1))
        all_sets_masks.append(mask)
        
    n_sets = len(all_sets_masks)
    
    # 2. Precompute req and forbid matrices
    req = np.zeros((n_sets, n_sets), dtype=np.int32)
    forbid = np.zeros((n_sets, n_sets), dtype=np.int32)
    
    for i in range(n_sets):
        mi = all_sets_masks[i]
        for j in range(i + 1, n_sets):
            mj = all_sets_masks[j]
            r = mi & mj
            f = mi ^ mj
            req[i][j] = r
            req[j][i] = r
            forbid[i][j] = f
            forbid[j][i] = f
            
    # Precompute triu indices for sizes 2 to 250
    triu_cache = {}
    for k in range(2, 251):
        triu_cache[k] = np.triu_indices(k, k=1)

    # 3. Greedy construction function using Numpy for conflict checking
    def _greedy(order, deadline):
        family = []
        family_np = np.array([], dtype=np.int32)
        
        for c in order:
            if time.time() > deadline:
                break
            
            n_f = len(family)
            if n_f >= 2:
                mc = all_sets_masks[c]
                
                i_idx, j_idx = triu_cache[n_f]
                i_arr = family_np[i_idx]
                j_arr = family_np[j_idx]
                
                r_arr = req[i_arr, j_arr]
                f_arr = forbid[i_arr, j_arr]
                
                mask_r = (mc & r_arr) == r_arr
                mask_f = (mc & f_arr) == 0
                
                if not np.any(mask_r & mask_f):
                    family.append(c)
                    family_np = np.append(family_np, c)
            else:
                family.append(c)
                family_np = np.append(family_np, c)
                
        return family

    best_family = []
    best_size = 0
    rng = random.Random(42)
    
    # Strategy: Multiple random restarts
    while time.time() < deadline - 5:
        order = list(range(n_sets))
        rng.shuffle(order)
        
        family = _greedy(order, deadline - 5)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            
            if best_size >= 250:
                break

    # Convert masks back to lists of elements
    result = []
    for idx in best_family:
        mask = all_sets_masks[idx]
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(elements)
        
    return result
# EVOLVE-BLOCK-END
