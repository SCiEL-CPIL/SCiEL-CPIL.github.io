import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10:
        return [11, 6]

    best_err = float('inf')
    best_pair = None
    
    # 2. Vectorized search for small k = 2a - 3b
    # Error = |k| / (2b). We want small |k| and large b.
    # We iterate k from 1 upwards. For each k, we find the largest b 
    # such that a = (3b+k)/2 is an integer, a <= N, and sigma(a) == sigma(b).
    
    # We can go up to k=1000. 
    # For k=1000, min error is 1000 / (2*10^7) = 5e-5.
    # For k=1, if we find b=10^6, error is 5e-7.
    # So small k is preferred.
    
    max_k = 1000
    
    for k in range(1, max_k + 1):
        if time.time() > deadline - 5:
            break
            
        # Determine parity of b
        # 3b + k must be even.
        # If k is even, 3b is even => b is even.
        # If k is odd, 3b is odd => b is odd.
        
        if k % 2 == 0:
            # b must be even
            b_max = int((2 * N - k) / 3)
            if b_max % 2 != 0:
                b_max -= 1
            b_min = 2
            if b_max < b_min:
                continue
            b_vals = np.arange(b_min, b_max + 1, 2)
        else:
            # b must be odd
            b_max = int((2 * N - k) / 3)
            if b_max % 2 == 0:
                b_max -= 1
            b_min = 1
            if b_max < b_min:
                continue
            b_vals = np.arange(b_min, b_max + 1, 2)
            
        if len(b_vals) == 0:
            continue
            
        # a = (3b + k) / 2
        a_vals = (3 * b_vals + k) // 2
        
        # Check sigma(a) == sigma(b)
        # Ensure a_vals are within bounds (they should be by construction, but just in case)
        # a_vals max is (3*b_max + k)//2 <= (3*(2N-k)/3 + k)//2 = (2N-k+k)//2 = N
        # a_vals min is (3*1 + 1)//2 = 2 (for k=1, b=1) or similar.
        
        mask = sigma[a_vals] == sigma[b_vals]
        
        if np.any(mask):
            # Find the largest b. Since b_vals is sorted ascending, 
            # the last index with mask=True is the largest b.
            idx = np.where(mask)[0]
            if len(idx) > 0:
                best_idx = idx[-1]
                b_best = int(b_vals[best_idx])
                a_best = int(a_vals[best_idx])
                
                # Calculate exact error
                # err = |a/b - 1.5| = |2a - 3b| / (2b) = |k| / (2b)
                err = float(k) / (2.0 * b_best)
                
                if err < best_err:
                    best_err = err
                    best_pair = (a_best, b_best)
                    
    if best_pair is not None:
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
