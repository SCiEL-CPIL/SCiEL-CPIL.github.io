import sys
import math
import time
from collections import defaultdict
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin for 90s limit
    
    MAX_N = 10_000_000
    
    # Build SPF (smallest prime factor) sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute sigma using multiplicative property with SPF
    # Also track the "remaining" part after removing all factors of spf[n]
    # sigma[n] = sigma[spf[n]^k] * sigma[n / spf[n]^k]
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    # rem[n] = n with all factors of spf[n] removed
    rem = [0] * (MAX_N + 1)
    rem[1] = 1
    # spow[n] = highest power of spf[n] dividing n
    spow = [1] * (MAX_N + 1)
    
    for n in range(2, MAX_N + 1):
        if time.time() > deadline - 15:
            break
        p = spf[n]
        if n % p == 0:
            prev = n // p
            if prev % p == 0:
                # p divides prev, so spow[n] = spow[prev] * p
                spow[n] = spow[prev] * p
                rem[n] = rem[prev]
                # sigma[n] = sigma[prev] * p + sigma[rem[n]]
                # Actually: sigma[p^k * r] = (1+p+...+p^k) * sigma(r)
                # sigma[prev] = (1+p+...+p^(k-1)) * sigma(r)
                # sigma[n] = (1+p+...+p^k) * sigma(r) = sigma[prev] + p^k * sigma(r)
                # p^k = spow[n], sigma(r) = sigma[rem[n]]
                sigma[n] = sigma[prev] + spow[n] * sigma[rem[n]]
            else:
                # p does not divide prev, so spow[n] = p, rem[n] = prev
                spow[n] = p
                rem[n] = prev
                sigma[n] = (1 + p) * sigma[prev]
        else:
            # This shouldn't happen since spf[n] always divides n
            spow[n] = p
            rem[n] = n // p
            sigma[n] = (1 + p) * sigma[rem[n]]
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # Group numbers by sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    best_err_num = float('inf')
    best_err_den = 1
    best_a = 11
    best_b = 6
    
    # For each sigma group, find pairs (a, b) minimizing |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # For each b in the group, find a in the group closest to 1.5*b
    for s_val, nums in sigma_groups.items():
        if time.time() > deadline - 2:
            break
        if len(nums) < 2:
            continue
        
        nums_sorted = sorted(nums)  # Already in order since we iterate 1..MAX_N
        
        for b in nums_sorted:
            target_a = 1.5 * b
            
            # Binary search for position of target_a in nums_sorted
            pos = bisect.bisect_left(nums_sorted, target_a)
            
            # Check candidates around pos
            candidates = set()
            if pos < len(nums_sorted):
                candidates.add(nums_sorted[pos])
            if pos > 0:
                candidates.add(nums_sorted[pos - 1])
            # Also check a few neighbors for safety
            for delta in range(-3, 4):
                idx = pos + delta
                if 0 <= idx < len(nums_sorted):
                    candidates.add(nums_sorted[idx])
            
            for a in candidates:
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # Invalid: error is exactly 0
                den = 2 * b
                
                # Compare num/den with best_err_num/best_err_den using cross multiplication
                if num * best_err_den < best_err_num * den:
                    best_err_num = num
                    best_err_den = den
                    best_a = a
                    best_b = b
    
    if best_err_num == float('inf'):
        return [11, 6]
    
    return [int(best_a), int(best_b)]
