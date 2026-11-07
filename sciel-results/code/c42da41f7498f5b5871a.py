import sys
import time
import math
import numpy as np
from bisect import bisect_left

# EVOLVE-BLOCK-START
MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Compute sigma(n) using NumPy
    # sigma[n] = sum of divisors of n
    # We can do this by adding i to all multiples of i.
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Looping 10^7 times in Python is ~1s.
    # Each step does a vectorized add.
    # Total operations ~ N log N.
    for i in range(1, MAX_N + 1):
        if time.time() > deadline - 10: # Reserve 10s for search
            # Fallback: use a smaller N or just return a known good pair?
            # If we can't compute sigma, we can't search.
            # But 10^7 loop is fast.
            pass
        sigma[i::i] += i

    # 2. Group numbers by sigma value
    # sigma values can be large. Max sigma(n) for n<=10^7 is around 4-5 * 10^7.
    # Using a dictionary is efficient.
    sigma_groups = {}
    
    # Convert numpy array to list for faster iteration? Or iterate numpy array.
    # Iterating numpy array is fast.
    # sigma is int64.
    for n in range(1, MAX_N + 1):
        s = sigma[n]
        if s in sigma_groups:
            sigma_groups[s].append(n)
        else:
            sigma_groups[s] = [n]

    # 3. Search for best pair
    best_a, best_b = 11, 6
    best_err_num, best_err_den = 1, 3 # |11/6 - 3/2| = |22/12 - 18/12| = 4/12 = 1/3
    best_score_val = 1/3.0

    # Iterate through all groups
    for s, group in sigma_groups.items():
        if len(group) < 2:
            continue
        
        # Sort the group to allow binary search for a ~ 1.5*b
        group.sort()
        
        # For each b in the group, find best a
        # We can iterate b and use bisect.
        # To optimize, we can just iterate.
        for b in group:
            target = 1.5 * b
            
            # Find position in sorted group
            pos = bisect_left(group, target)
            
            # Check pos and pos-1
            # pos is the index where target would be inserted to maintain order.
            # So group[pos] >= target, group[pos-1] < target.
            
            candidates = []
            if pos < len(group):
                candidates.append(group[pos])
            if pos > 0:
                candidates.append(group[pos-1])
            
            for a in candidates:
                if a == b:
                    continue
                
                # Check if a is within valid range (it is, from 1 to MAX_N)
                # Calculate error |a/b - 3/2| = |2a - 3b| / (2b)
                # Numerator: |2a - 3b|
                # Denominator: 2b
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                
                den = 2 * b
                
                # Compare num/den with best_err_num/best_err_den
                # num * best_err_den < best_err_num * den
                if num * best_err_den < best_err_num * den:
                    best_a, best_b = a, b
                    best_err_num, best_err_den = num, den
                    best_score_val = num / den
                    
                    # If we find an error of 1/(2b) for a large b, it's very good.
                    # Can we break early?
                    # The theoretical minimum error for a given b is 1/(2b) if 3b is odd.
                    # If we find num=1, it's the best possible for this b.
                    # But we still need to check other b's which might be larger.
                    # However, if we find num=1 for a large b, it's a very strong candidate.
                    # We can't easily prune the outer loop without knowing max b remaining.
                    # So we just continue.

    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
