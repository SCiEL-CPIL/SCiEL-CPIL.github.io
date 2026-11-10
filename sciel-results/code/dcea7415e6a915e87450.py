import numpy as np
import time

def construct():
    deadline = time.time() + 85
    MAXN = 10_000_000
    
    # Compute sigma using sieve
    sigma = np.zeros(MAXN + 1, dtype=np.int64)
    for d in range(1, MAXN + 1):
        sigma[d::d] += d
    
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    W = 50
    chunk_size = 200_000
    
    for b_start in range(1, MAXN + 1, chunk_size):
        b_end = min(b_start + chunk_size, MAXN + 1)
        b_vals = np.arange(b_start, b_end, dtype=np.int64)
        
        # Even b
        b_even_mask = (b_vals % 2 == 0)
        if np.any(b_even_mask):
            b_even = b_vals[b_even_mask]
            for offset in range(-W, W + 1):
                a = (3 * b_even + 2 * offset) // 2
                valid = (a >= 1) & (a <= MAXN)
                if not np.any(valid):
                    continue
                a_v = a[valid]
                b_v = b_even[valid]
                matches = sigma[a_v] == sigma[b_v]
                if np.any(matches):
                    a_m = a_v[matches]
                    b_m = b_v[matches]
                    err = np.abs(2 * a_m - 3 * b_m) / (2.0 * b_m)
                    nz = err > 0
                    if np.any(nz):
                        e = err[nz]
                        am = a_m[nz]
                        bm = b_m[nz]
                        idx = np.argmin(e)
                        if e[idx] < best_err:
                            best_err = float(e[idx])
                            best_a = int(am[idx])
                            best_b = int(bm[idx])
        
        # Odd b
        b_odd_mask = (b_vals % 2 == 1)
        if np.any(b_odd_mask):
            b_odd = b_vals[b_odd_mask]
            for offset in range(-W, W + 1):
                a1 = (3 * b_odd - 1) // 2 + offset
                a2 = (3 * b_odd + 1) // 2 + offset
                for a_arr in (a1, a2):
                    valid = (a_arr >= 1) & (a_arr <= MAXN)
                    if not np.any(valid):
                        continue
                    a_v = a_arr[valid]
                    b_v = b_odd[valid]
                    matches = sigma[a_v] == sigma[b_v]
                    if np.any(matches):
                        a_m = a_v[matches]
                        b_m = b_v[matches]
                        err = np.abs(2 * a_m - 3 * b_m) / (2.0 * b_m)
                        nz = err > 0
                        if np.any(nz):
                            e = err[nz]
                            am = a_m[nz]
                            bm = b_m[nz]
                            idx = np.argmin(e)
                            if e[idx] < best_err:
                                best_err = float(e[idx])
                                best_a = int(am[idx])
                                best_b = int(bm[idx])
        
        if time.time() > deadline - 5:
            break
    
    return [int(best_a), int(best_b)]
