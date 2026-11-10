import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 78
    
    D = 2329089562800  # LCM(1..30)
    half = 15
    
    w1 = [D // (i+1) for i in range(half)]
    w2 = [D // (i+1) for i in range(half, 30)]
    
    sums1 = np.array([0], dtype=np.int64)
    for w in w1:
        sums1 = np.concatenate((sums1 - w, sums1, sums1 + w))
    
    if time.time() - start_time > DEADLINE:
        return [0]*28 + [1, -1]
    
    sums2 = np.array([0], dtype=np.int64)
    for w in w2:
        sums2 = np.concatenate((sums2 - w, sums2, sums2 + w))
    
    if time.time() - start_time > DEADLINE:
        return [0]*28 + [1, -1]
    
    order2 = np.argsort(sums2)
    sorted_sums2 = sums2[order2]
    del sums2
    
    if time.time() - start_time > DEADLINE:
        return [0]*28 + [1, -1]
    
    targets = -sums1
    positions = np.searchsorted(sorted_sums2, targets)
    
    pos1 = np.clip(positions, 0, len(sorted_sums2)-1)
    pos2 = np.clip(positions - 1, 0, len(sorted_sums2)-1)
    
    cand1 = sums1 + sorted_sums2[pos1]
    cand2 = sums1 + sorted_sums2[pos2]
    
    abs_cand1 = np.abs(cand1).copy()
    abs_cand2 = np.abs(cand2).copy()
    abs_cand1[cand1 == 0] = np.iinfo(np.int64).max
    abs_cand2[cand2 == 0] = np.iinfo(np.int64).max
    
    best_abs = np.minimum(abs_cand1, abs_cand2)
    best_idx = int(np.argmin(best_abs))
    
    if time.time() - start_time > DEADLINE:
        return [0]*28 + [1, -1]
    
    if abs_cand1[best_idx] <= abs_cand2[best_idx]:
        orig_i = best_idx
        orig_j = int(order2[pos1[best_idx]])
    else:
        orig_i = best_idx
        orig_j = int(order2[pos2[best_idx]])
    
    coeffs = [0] * 30
    temp = orig_i
    for k in range(half):
        d = temp % 3
        temp //= 3
        coeffs[k] = d - 1
    
    temp = orig_j
    for k in range(half):
        d = temp % 3
        temp //= 3
        coeffs[half + k] = d - 1
    
    total_num = sum(coeffs[i] * (D // (i+1)) for i in range(30))
    if total_num == 0:
        return [0]*28 + [1, -1]
    
    return coeffs
# EVOLVE-BLOCK-END
