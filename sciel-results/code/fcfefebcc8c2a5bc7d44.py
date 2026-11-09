import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 10_000_000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 5:
        return [11, 6]

    # Group numbers by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)
        
    if time.time() > deadline - 2:
        return [11, 6]

    best_err = float('inf')
    best_pair = None
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        
        if time.time() > deadline - 1:
            break
            
        nums_sorted = sorted(nums)
        
        # Two-pointer search
        i = 0
        n_len = len(nums_sorted)
        
        for j in range(n_len):
            b = nums_sorted[j]
            target = 1.5 * b
            
            # Advance i to be the first index where nums_sorted[i] >= target
            while i < n_len and nums_sorted[i] < target:
                i += 1
                
            # Check i and i-1
            if i > 0:
                a = nums_sorted[i-1]
                num = abs(2*a - 3*b)
                if num != 0:
                    den = 2*b
                    err = num / den
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        
            if i < n_len:
                a = nums_sorted[i]
                num = abs(2*a - 3*b)
                if num != 0:
                    den = 2*b
                    err = num / den
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        
    if best_pair is not None:
        return [int(best_pair[0]), int(best_pair[1])]
    else:
        return [11, 6]
