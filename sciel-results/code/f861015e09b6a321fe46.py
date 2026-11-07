import math
import time
import bisect
import array
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX = 10000000
    
    # 1. Sieve for smallest prime factor (SPF)
    # Use array('I') for 32-bit unsigned int, saves memory
    spf = array.array('I', range(MAX + 1))
    
    # Mark evens
    for i in range(4, MAX + 1, 2):
        spf[i] = 2
        
    limit = int(MAX**0.5) + 1
    for i in range(3, limit, 2):
        if spf[i] == i: # i is prime
            step = 2 * i
            start = i * i
            for j in range(start, MAX + 1, step):
                if spf[j] == j:
                    spf[j] = i
                    
    if time.time() > deadline:
        return [11, 6]

    # 2. Compute sigma(n)
    # sigma is multiplicative.
    # Use array('Q') for 64-bit unsigned int to hold sigma values (can be large)
    sigma = array.array('Q', [0]) * (MAX + 1)
    # power[n] stores p^k where p=spf[n]
    power = array.array('Q', [0]) * (MAX + 1)
    
    sigma[1] = 1
    power[1] = 1
    
    for i in range(2, MAX + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            pk = power[m] * p
            power[i] = pk
            r = m // power[m]
            sigma[i] = sigma[m] + pk * sigma[r]
        else:
            power[i] = p
            sigma[i] = sigma[m] * (p + 1)
            
        if i % 1000000 == 0 and time.time() > deadline:
            return [11, 6]

    if time.time() > deadline:
        return [11, 6]

    # 3. Group numbers by sigma value
    groups = {}
    for i in range(1, MAX + 1):
        s = sigma[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    if time.time() > deadline:
        return [11, 6]

    # 4. Search for best pair
    best_a = 11
    best_b = 6
    best_num = 4 # |2*11 - 3*6|
    best_den = 12 # 2*6
    
    # Sort groups by size descending? No, just iterate.
    # Large groups are more likely to have good pairs, but small ones might have the global min.
    # We must check all.
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        if time.time() > deadline:
            break
            
        nums.sort()
        K = len(nums)
        
        # Optimization: Use bisect for each b
        # Or two-pointer?
        # Let's use bisect as it's C-implemented and robust.
        
        for b in nums:
            target_a = 1.5 * b
            idx = bisect.bisect_left(nums, target_a)
            
            # Check idx
            if idx < K:
                a = nums[idx]
                if a != b:
                    diff = abs(2 * a - 3 * b)
                    den = 2 * b
                    if diff * best_den < best_num * den:
                        best_num = diff
                        best_den = den
                        best_a = a
                        best_b = b
                        
            # Check idx - 1
            if idx > 0:
                a = nums[idx - 1]
                if a != b:
                    diff = abs(2 * a - 3 * b)
                    den = 2 * b
                    if diff * best_den < best_num * den:
                        best_num = diff
                        best_den = den
                        best_a = a
                        best_b = b
                        
    return [int(best_a), int(best_b)]
