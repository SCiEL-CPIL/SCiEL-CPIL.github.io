import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin
    
    N = 10000000
    
    # Compute sigma using SPF sieve
    spf = np.zeros(N + 1, dtype=np.int64)
    primes = []
    for i in range(2, N + 1):
        if time.time() > deadline - 40.0:
            return [11, 6]
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p
    
    if time.time() > deadline - 35.0:
        return [11, 6]
    
    pk = np.zeros(N + 1, dtype=np.int64)
    rest = np.zeros(N + 1, dtype=np.int64)
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    pk[1] = 1
    rest[1] = 1
    
    for i in range(2, N + 1):
        if time.time() > deadline - 30.0:
            return [11, 6]
        p = spf[i]
        m = i // p
        if m % p == 0:
            pk[i] = pk[m] * p
            rest[i] = rest[m]
        else:
            pk[i] = p
            rest[i] = m
        sigma[i] = sigma[rest[i]] * (pk[i] * p - 1) // (p - 1)
    
    if time.time() > deadline - 25.0:
        return [11, 6]
    
    # Group by sigma value
    groups = {}
    for n in range(1, N + 1):
        if time.time() > deadline - 15.0:
            break
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
    
    if time.time() > deadline - 10.0:
        return [11, 6]
    
    # Find best pair
    best_val = 1.0 / 3.0  # |11/6 - 3/2| = 1/3
    best_pair = (11, 6)
    
    for s, nums in groups.items():
        if time.time() > deadline - 5.0:
            break
        k = len(nums)
        if k < 2:
            continue
        # nums is already sorted
        for j in range(k):
            b = nums[j]
            target = 1.5 * b
            pos = bisect_left(nums, target)
            for i in (pos - 1, pos):
                if 0 <= i < k:
                    a = nums[i]
                    if a == b:
                        continue
                    val = abs(a / b - 1.5)
                    if val < best_val:
                        best_val = val
                        best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]
