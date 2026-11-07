import sys
import time
import math
import numpy as np
from collections import defaultdict
from bisect import bisect_left, bisect_right

MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Compute sigma(n) for all n <= MAX_N using numpy
    # sigma = [0] * (MAX_N + 1) is too slow in pure python for the update loop
    # Use numpy array
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve
    # for i in range(1, MAX_N + 1):
    #     sigma[i::i] += i
    # This loop in python is slow. 
    # We can try to optimize it or just use it if numpy vectorization is fast enough.
    # Actually, the loop itself is 10^7 iterations. In Python, 10^7 iterations takes ~0.5-1s.
    # The inner numpy operation `sigma[i::i] += i` is fast.
    # Total time should be well under 10s.
    
    for i in range(1, MAX_N + 1):
        if time.time() > deadline - 10: # Safety margin
            break
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [11, 6]

    # 2. Group indices by sigma value
    # We only care about sigma values that appear at least twice?
    # Actually, we need to find pairs.
    # Creating a dict of lists for 10^7 items is memory heavy but doable.
    # Max sigma value is roughly N log log N ~ 10^7 * 3 ~ 3*10^7.
    # Dict size will be up to 3*10^7 keys? No, many sigma values are unique.
    # Number of unique sigma values is less than N.
    # Let's just build the dict.
    
    sigma_groups = defaultdict(list)
    # Iterating 10^7 items in Python is ~1s.
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
        
    if time.time() > deadline:
        return [11, 6]

    # 3. Search for best pair
    best_a, best_b = 11, 6
    best_err_num, best_err_den = 4, 12 # |11/6 - 3/2| = |22-18|/12 = 4/12 = 1/3
    best_score_val = 1.0 / 3.0
    
    # We want to minimize |2a - 3b| / (2b)
    # Iterate through groups
    for s_val, L in sigma_groups.items():
        if time.time() > deadline:
            break
            
        if len(L) < 2:
            continue
            
        # L is already sorted
        # For each b in L, find a in L closest to 1.5 * b
        # To avoid O(N^2), we can iterate b and binary search a.
        # But if a group is large, this is expensive.
        # Most groups are small.
        
        # Optimization: We only care if potential error is better than current best.
        # Min possible error for this b is ~ 1/(2b) if we can find integer a.
        # Actually, we are restricted to a in L.
        
        for b in L:
            if time.time() > deadline:
                break
                
            # Pruning: if 1/(2*b) >= best_score_val, and b is small, maybe skip?
            # No, we iterate all b. 
            # But if b is very small, 1/(2b) is large.
            # We want small error.
            
            target = 1.5 * b
            # Find a in L closest to target
            idx = bisect_left(L, target)
            
            candidates = []
            if idx < len(L):
                candidates.append(L[idx])
            if idx > 0:
                candidates.append(L[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                # Check sigma(a) == sigma(b) is guaranteed by group
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # Invalid score 0
                    continue
                den = 2 * b
                # Compare num/den with best_err_num/best_err_den
                if num * best_err_den < best_err_num * den:
                    best_a, best_b = a, b
                    best_err_num, best_err_den = num, den
                    best_score_val = num / float(den)

    return [int(best_a), int(best_b)]
