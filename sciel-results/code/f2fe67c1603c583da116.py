import numpy as np

def construct():
    N = 10_000_000
    
    # Compute sigma(n) for all n from 1 to N using sieve
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    # For each b, candidate a values are floor(1.5*b) and ceil(1.5*b)
    b_vals = np.arange(1, N + 1, dtype=np.int64)
    
    a_floor = (3 * b_vals) // 2
    a_ceil = a_floor + 1
    
    # Valid ranges
    mask_floor = (a_floor >= 1) & (a_floor <= N)
    mask_ceil = (a_ceil >= 1) & (a_ceil <= N)
    
    # Compute errors and check sigma equality
    bf = b_vals[mask_floor]
    af = a_floor[mask_floor]
    err_f = np.abs(af.astype(np.float64) / bf.astype(np.float64) - 1.5)
    valid_f = sigma[af] == sigma[bf]
    
    bc = b_vals[mask_ceil]
    ac = a_ceil[mask_ceil]
    err_c = np.abs(ac.astype(np.float64) / bc.astype(np.float64) - 1.5)
    valid_c = sigma[ac] == sigma[bc]
    
    # Combine
    all_err = np.concatenate([err_f[valid_f], err_c[valid_c]])
    if len(all_err) == 0:
        return [11, 6]
    
    min_idx = np.argmin(all_err)
    
    n_f = len(err_f[valid_f])
    if min_idx < n_f:
        idx = np.where(valid_f)[0][min_idx]
        a = int(af[idx])
        b = int(bf[idx])
    else:
        idx = np.where(valid_c)[0][min_idx - n_f]
        a = int(ac[idx])
        b = int(bc[idx])
    
    error = abs(float(a) / float(b) - 1.5)
    if error == 0.0:
        # Find next best non-zero error
        mask_nonzero = all_err > 0
        if np.any(mask_nonzero):
            # Set zero errors to inf and find min
            adjusted = np.where(mask_nonzero, all_err, np.inf)
            min_idx2 = np.argmin(adjusted)
            if min_idx2 < n_f:
                idx = np.where(valid_f)[0][min_idx2]
                a = int(af[idx])
                b = int(bf[idx])
            else:
                idx = np.where(valid_c)[0][min_idx2 - n_f]
                a = int(ac[idx])
                b = int(bc[idx])
        else:
            return [11, 6]
    
    return [a, b]
