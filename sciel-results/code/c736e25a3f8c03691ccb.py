import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    DEADLINE = 75.0
    
    N = 10_000_000
    
    # Step 1: Sieve sigma(n) for n=1..N using NumPy
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15:
        return [11, 6]
    
    # Step 2: Group by sigma value
    sigma_arr = sigma[1:]  # corresponds to n=1..N
    order = np.argsort(sigma_arr, kind='stable')
    sorted_sigma = sigma_arr[order]
    sorted_indices = order + 1  # n values (1-indexed)
    
    # Find group boundaries
    diff = np.diff(sorted_sigma)
    changes = np.where(diff != 0)[0] + 1
    boundaries = np.concatenate(([0], changes, [len(sorted_sigma)]))
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return [11, 6]
    
    # Step 3: Find best pair minimizing |a/b - 3/2|
    best_num = 1
    best_den = 1
    best_a, best_b = 11, 6  # fallback: sigma(11)=12=sigma(6)
    
    for i in range(len(boundaries) - 1):
        lo = int(boundaries[i])
        hi = int(boundaries[i + 1])
        if hi - lo < 2:
            continue
        
        if time.time() - start_time > DEADLINE - 3:
            break
        
        group = sorted_indices[lo:hi]
        group_len = len(group)
        
        for b_val in group:
            b_int = int(b_val)
            # Ideal a = 3b/2
            target = (3 * b_int) // 2
            
            pos = bisect.bisect_left(group, target)
            
            # Check candidates around pos
            lo_idx = max(0, pos - 2)
            hi_idx = min(group_len, pos + 3)
            
            for idx in range(lo_idx, hi_idx):
                a_int = int(group[idx])
                if a_int == b_int:
                    continue
                num = abs(2 * a_int - 3 * b_int)
                if num == 0:
                    continue
                den = 2 * b_int
                # Compare num/den < best_num/best_den  <=>  num*best_den < best_num*den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a, best_b = a_int, b_int
    
    # Ensure Python int types
    best_a = int(best_a)
    best_b = int(best_b)
    
    return [best_a, best_b]
