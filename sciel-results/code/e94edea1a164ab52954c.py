"""Combined program: SPF-based sigma computation + group search for divisor-sum fiber pairs."""
import time
import math
import bisect
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
MAX_N = 10_000_000

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin within 90s limit
    
    # Step 1: Compute sigma(n) for all n up to MAX_N using SPF sieve
    # Build smallest prime factor array
    spf = np.arange(MAX_N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    
    # Sieve for smallest prime factors
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma(n) multiplicatively using SPF
    # For n = p1^a1 * p2^a2 * ... * pk^ak
    # sigma(n) = prod_i (p_i^(a_i+1) - 1) / (p_i - 1)
    # We can compute this incrementally:
    # Let n = m * p where p = spf[n]
    # If p divides m, then n has a higher power of p than m
    # sigma(n) = sigma(m) * (p^(a+1)-1)/(p-1) / ((p^a-1)/(p-1)) where a is power of p in m
    # Simpler: track for each n, the "p-part" contribution
    
    # Alternative efficient approach:
    # sigma[n] can be computed from sigma of n/p and the power of p dividing n
    # Let's use: for each n, factor it using spf and compute sigma directly
    # But that's O(log n) per n, total O(n log n) which is fine
    
    # Better: iterative DP using spf
    # sigma[1] = 1
    # For n > 1:
    #   p = spf[n]
    #   m = n // p
    #   If m % p == 0:
    #     Let p^k || m (p^k divides m but p^(k+1) doesn't)
    #     Then p^(k+1) || n
    #     sigma(n) = sigma(m) - sigma(m / p^k) + (p^(k+2)-1)/(p-1) ... this gets complex
    #   
    # Cleanest: for each n, compute sigma by factoring
    # sigma[n] = product over prime powers p^a || n of (p^(a+1)-1)/(p-1)
    
    # Let me use a different efficient method:
    # Precompute for each n: the largest divisor d of n that is a power of spf[n]
    # Then sigma[n] = sigma[n/d] * (d*p - 1)/(p - 1) where p = spf[n]
    
    # Actually, the simplest correct approach for n up to 10^7:
    # Use the fact that sigma is multiplicative and compute via SPF
    # For each n:
    #   p = spf[n]
    #   m = n // p
    #   Count how many times p divides n: let that be k
    #   n = p^k * r where gcd(r, p) = 1
    #   sigma(n) = sigma(r) * (p^(k+1) - 1) / (p - 1)
    
    # Precompute power contributions
    # For efficiency, let's compute sigma incrementally:
    # sigma[1] = 1
    # For n >= 2:
    #   p = spf[n]
    #   m = n // p
    #   If m % p == 0:
    #     # p divides m, so the p-adic valuation increases
    #     # We need to know the full p-power in n
    #     # Let's precompute for each n: p_powers[n] = largest power of spf[n] dividing n
    #   This is getting complicated. Let me just do direct factorization per n.
    
    # Direct approach: for each n, factor using spf and compute sigma
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    sigma[1] = 1
    
    for n in range(2, MAX_N + 1):
        if time.time() > deadline - 20:
            break
        p = spf[n]
        m = n
        s_val = 1
        while m > 1:
            p = spf[m]
            # Count power of p in m
            power = 0
            p_power = 1
            while m % p == 0:
                m //= p
                power += 1
                p_power *= p
            # sigma contribution: (p^(power+1) - 1) / (p - 1)
            # = 1 + p + p^2 + ... + p^power
            p_sigma = (p_power * p - 1) // (p - 1)
            s_val *= p_sigma
        sigma[n] = s_val
    
    if time.time() > deadline - 15:
        return [11, 6]
    
    # Step 2: Group numbers by sigma value
    sigma_list = sigma.tolist()
    
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma_list[n]].append(n)
        if time.time() > deadline - 10:
            break
    
    # Step 3: For each group, find pair (a,b) minimizing |a/b - 3/2|
    best_error = float('inf')
    best_pair = (11, 6)
    
    # Sort groups by size descending to find good pairs faster
    # But we need to check all groups. Let's process all.
    
    for s, nums in sigma_groups.items():
        if time.time() > deadline - 5:
            break
        if len(nums) < 2:
            continue
        
        nums_sorted = sorted(nums)
        n_len = len(nums_sorted)
        
        # For each b, find a closest to 3b/2
        # We want to minimize |2a - 3b| / (2b)
        # For large b, even small |2a-3b| gives small error
        # Prioritize checking larger b values
        
        # Use two-pointer or binary search
        # For each b, target a = 3b/2
        # Binary search in nums_sorted
        
        for b in nums_sorted:
            target_a = 3 * b / 2.0
            idx = bisect.bisect_left(nums_sorted, int(target_a))
            
            # Check candidates around idx
            for i in range(max(0, idx - 3), min(n_len, idx + 4)):
                a = nums_sorted[i]
                if a == b:
                    continue
                numerator = abs(2 * a - 3 * b)
                if numerator == 0:
                    continue
                denominator = 2 * b
                error = numerator / denominator
                if error < best_error:
                    best_error = error
                    best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
