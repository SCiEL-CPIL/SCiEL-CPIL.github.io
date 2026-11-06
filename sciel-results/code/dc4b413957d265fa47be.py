import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    lcm_val = 2329089562800
    
    def build_sums(start, end):
        sums = np.array([0], dtype=np.int64)
        indices = np.array([0], dtype=np.int64)
        
        for k in range(end - start + 1):
            i = start + k
            w = lcm_val // i
            m = 3 ** k
            
            s0 = sums
            i0 = indices
            s1 = sums + w
            i1 = indices + m
            s2 = sums - w
            i2 = indices + 2 * m
            
            all_sums = np.concatenate([s0, s1, s2])
            all_indices = np.concatenate([i0, i1, i2])
            
            sort_idx = np.argsort(all_sums, kind='stable')
            sums = all_sums[sort_idx]
            indices = all_indices[sort_idx]
            
        return sums, indices

    sums_A, idx_A = build_sums(1, 15)
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]

    sums_B, idx_B = build_sums(16, 30)
    if time.time() > deadline - 5:
        return [0] * 28 + [1, -1]

    best_diff = float('inf')
    best_idx_A = 0
    best_idx_B = 0
    
    # sums_B is sorted
    for i in range(len(sums_A)):
        if time.time() > deadline - 2:
            break
            
        s_A = sums_A[i]
        target = -s_A
        
        pos = np.searchsorted(sums_B, target, side='left')
        
        # Check pos-1, pos, pos+1
        for j in [int(pos)-1, int(pos), int(pos)+1]:
            if 0 <= j < len(sums_B):
                s_B = sums_B[j]
                diff = abs(s_A + s_B)
                if diff > 0 and diff < best_diff:
                    best_diff = diff
                    best_idx_A = idx_A[i]
                    best_idx_B = idx_B[j]
                    if best_diff == 1:
                        break
        if best_diff == 1:
            break
            
    if best_diff == float('inf'):
        return [0] * 28 + [1, -1]

    coeffs = [0] * 30
    
    temp_idx = int(best_idx_A)
    for k in range(15):
        d = (temp_idx // (3 ** k)) % 3
        if d == 0:
            coeffs[k] = 0
        elif d == 1:
            coeffs[k] = 1
        else:
            coeffs[k] = -1
            
    temp_idx = int(best_idx_B)
    for k in range(15):
        d = (temp_idx // (3 ** k)) % 3
        if d == 0:
            coeffs[15 + k] = 0
        elif d == 1:
            coeffs[15 + k] = 1
        else:
            coeffs[15 + k] = -1
            
    return coeffs
