import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N using vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [1, 1]

    # 2. Vectorized bucketing: sort indices by sigma value
    idx = np.arange(1, N + 1)
    sigma_vals = sigma[1:]
    order = np.argsort(sigma_vals, kind='stable')
    sorted_idx = idx[order]
    sorted_sigma = sigma_vals[order]
    
    # Find group boundaries where sigma value changes
    changes = np.where(np.diff(sorted_sigma) != 0)[0] + 1
    group_starts = np.concatenate([[0], changes])
    group_ends = np.concatenate([changes, [len(sorted_idx)]])
    
    if time.time() > deadline:
        for i in range(len(group_starts)):
            if group_ends[i] - group_starts[i] >= 2:
                a = int(sorted_idx[group_starts[i]])
                b = int(sorted_idx[group_starts[i] + 1])
                return [a, b]
        return [1, 1]

    # 3. Search for best ratio within each group
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    for i in range(len(group_starts)):
        if time.time() > deadline:
            break
            
        s = group_starts[i]
        e = group_ends[i]
        k = e - s
        if k < 2:
            continue
        
        group_vals = np.sort(sorted_idx[s:e])
        
        b_vals = group_vals
        targets = 1.5 * b_vals
        
        insert_pts = np.searchsorted(group_vals, targets, side='left')
        
        cand1 = np.clip(insert_pts, 0, k - 1)
        cand0 = np.clip(insert_pts - 1, 0, k - 1)
        
        a_cand1 = group_vals[cand1]
        a_cand0 = group_vals[cand0]
        
        err1 = np.abs(2 * a_cand1 - 3 * b_vals).astype(np.float64) / (2.0 * b_vals)
        err0 = np.abs(2 * a_cand0 - 3 * b_vals).astype(np.float64) / (2.0 * b_vals)
        
        mask1 = (a_cand1 != b_vals)
        mask0 = (a_cand0 != b_vals)
        
        err1[~mask1] = np.inf
        err0[~mask0] = np.inf
        
        min_err1 = np.min(err1)
        min_err0 = np.min(err0)
        
        if min_err1 < min_err0:
            min_err = min_err1
            b_idx = np.argmin(err1)
            a_val = a_cand1[b_idx]
            b_val = b_vals[b_idx]
        else:
            min_err = min_err0
            b_idx = np.argmin(err0)
            a_val = a_cand0[b_idx]
            b_val = b_vals[b_idx]
        
        if min_err < best_err:
            best_err = min_err
            best_a = int(a_val)
            best_b = int(b_val)

    if best_err == float('inf'):
        # Fallback: find any valid pair
        for i in range(len(group_starts)):
            if group_ends[i] - group_starts[i] >= 2:
                best_a = int(sorted_idx[group_starts[i]])
                best_b = int(sorted_idx[group_starts[i] + 1])
                break
    
    return [best_a, best_b]
