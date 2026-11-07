import time
import numpy as np
from fractions import Fraction
from collections import defaultdict
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # Compute sigma(n) for all n up to N using O(N log N) sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        if time.time() > deadline - 30.0:
            break
        sigma[i::i] += i
    
    if time.time() > deadline - 30.0:
        return [11, 6]
    
    # Group numbers by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[int(sigma[n])].append(n)
    
    # For each group, find the pair (a,b) minimizing |2a - 3b|
    best_error = Fraction(1, 1)
    best_pair = (11, 6)
    
    # Process groups
    for s, nums in groups.items():
        if time.time() > deadline - 5.0:
            break
        if len(nums) < 2:
            continue
        
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        
        # For each b in nums_sorted, find a in nums_sorted closest to 3b/2
        for i in range(n):
            b = nums_sorted[i]
            target = 3 * b / 2.0
            
            idx = bisect_left(nums_sorted, target)
            
            candidates = []
            if idx < n:
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)
            
            for j in candidates:
                a = nums_sorted[j]
                if a == b:
                    continue
                delta = 2 * a - 3 * b
                if delta == 0:
                    continue
                err = Fraction(abs(delta), 2 * b)
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
    
    return [int(best_pair[0]), int(best_pair[1])]
