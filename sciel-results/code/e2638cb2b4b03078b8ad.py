import time
import numpy as np
from bisect import bisect_left
from collections import defaultdict

def construct():
    t_start = time.time()
    deadline = t_start + 85.0
    
    MAX_N = 10_000_000
    
    # Step 1: Compute smallest prime factor using linear sieve
    spf = np.zeros(MAX_N + 1, dtype=np.int32)
    primes = []
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > MAX_N:
                break
            spf[i * p] = p
    
    if time.time() > deadline - 40:
        return [11, 6]
    
    # Step 2: Compute sigma(n) using multiplicative property
    # sigma(n) = prod over prime powers p^k || n of (p^(k+1)-1)/(p-1)
    # We compute this iteratively using spf.
    
    sigma = np.ones(MAX_N + 1, dtype=np.int64)
    # Also track the "sigma contribution" of the largest prime power in n
    # For n, let p = spf[n], and let n = p^k * m where gcd(m, p) = 1
    # sigma(n) = sigma(p^k) * sigma(m)
    # We can compute sigma(p^k) on the fly.
    
    # Let's define: for each n, we want sigma[n]
    # If n = 1: sigma[1] = 1
    # If n > 1: let p = spf[n], m = n // p
    #   If m % p != 0: n = p * m, gcd(p, m) = 1
    #     sigma[n] = sigma[m] * (1 + p)
    #   If m % p == 0: n = p^k * r where k >= 2
    #     We need to know the full p-power. Let's track it.
    
    # Better approach: for each n, maintain:
    #   sigma[n] = full sigma of n
    #   ppow[n] = the largest power of spf[n] dividing n, i.e., p^k where p^k || n
    #   sigma_ppow[n] = sigma(p^k) = (p^(k+1)-1)/(p-1)
    
    # Then: if m = n // p, and m % p != 0:
    #   ppow[n] = p, sigma_ppow[n] = 1 + p
    #   sigma[n] = sigma[m] * sigma_ppow[n]
    # If m % p == 0:
    #   ppow[n] = ppow[m] * p
    #   sigma_ppow[n] = sigma_ppow[m] + ppow[n]  # since sigma(p^k) = sigma(p^(k-1)) + p^k
    #   sigma[n] = sigma[m // ppow[m]] * sigma_ppow[n]
    
    # Let me verify: sigma(p^k) = 1 + p + p^2 + ... + p^k
    # sigma(p^(k-1)) = 1 + p + ... + p^(k-1)
    # sigma(p^k) = sigma(p^(k-1)) + p^k ✓
    
    # And if n = p^k * r, gcd(p,r)=1, then sigma(n) = sigma(p^k) * sigma(r)
    # If m = n/p = p^(k-1) * r, then sigma(m) = sigma(p^(k-1)) * sigma(r)
    # So sigma(r) = sigma(m) / sigma(p^(k-1))
    # sigma(n) = sigma(p^k) * sigma(r) = sigma(p^k) * sigma(m) / sigma(p^(k-1))
    
    # This requires division which may not be exact in integers. Let me use the tracking approach.
    
    ppow = np.ones(MAX_N + 1, dtype=np.int64)  # p^k where p^k || n
    sigma_ppow = np.ones(MAX_N + 1, dtype=np.int64)  # sigma(p^k)
    
    for n in range(2, MAX_N + 1):
        p = spf[n]
        m = n // p
        if m % p != 0:
            # n = p * m, gcd(p, m) = 1
            ppow[n] = p
            sigma_ppow[n] = 1 + p
            sigma[n] = sigma[m] * sigma_ppow[n]
        else:
            # n = p^k * r, k >= 2
            ppow[n] = ppow[m] * p
            sigma_ppow[n] = sigma_ppow[m] + ppow[n]
            # sigma(n) = sigma(p^k) * sigma(r)
            # sigma(m) = sigma(p^(k-1)) * sigma(r)
            # So sigma(r) = sigma(m) // sigma_ppow[m]
            # sigma(n) = sigma_ppow[n] * (sigma(m) // sigma_ppow[m])
            sigma[n] = sigma_ppow[n] * (sigma[m] // sigma_ppow[m])
    
    if time.time() > deadline - 15:
        return [11, 6]
    
    # Step 3: Group numbers by sigma value
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        groups[s].append(n)
        
    if time.time() > deadline - 8:
        return [11, 6]
    
    # Step 4: Search for best pair
    best_err = float('inf')
    best_pair = (11, 6)
    
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    for s, nums in sorted_groups:
        if time.time() > deadline - 3:
            break
        if len(nums) < 2:
            continue
            
        n = len(nums)
        for i in range(n):
            b = nums[i]
            target = 1.5 * b
            pos = bisect_left(nums, target)
            
            start = max(0, pos - 100)
            end = min(n, pos + 101)
            
            for j in range(start, end):
                if j == i:
                    continue
                a = nums[j]
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue
                err = diff / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    
    a, b = best_pair
    return [int(a), int(b)]

