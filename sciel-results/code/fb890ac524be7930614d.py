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
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    
    for n in range(2, MAX_N + 1):
        if time.time() > deadline - 15:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            # n = p^k * r where p does not divide r
            pk = 1
            temp = n
            while temp % p == 0:
                pk *= p
                temp //= p
            sum_powers = (pk * p - 1) // (p - 1)
            sigma[n] = sum_powers * sigma[temp]
        else:
            sigma[n] = (1 + p) * sigma[m]
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # Group numbers by sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
    
    # Sort each group for binary search
    for key in sigma_groups:
        sigma_groups[key].sort()
    
    best_err_num = float('inf')
    best_err_den = 1
    best_a = 11
    best_b = 6
    
    # For each b, find closest a in same sigma group to 1.5*b
    for b in range(1, MAX_N + 1):
        if time.time() > deadline - 2:
            break
        
        sb = sigma[b]
        group = sigma_groups.get(sb)
        if group is None or len(group) < 2:
            continue
        
        target = 3 * b / 2.0
        
        # Binary search for closest to target
        idx = bisect.bisect_left(group, target)
        
        # Check candidates around idx
        candidates = set()
        for i in range(max(0, idx - 2), min(len(group), idx + 3)):
            if i < len(group):
                candidates.add(group[i])
        
        for a in candidates:
            if a == b:
                continue
            if a > MAX_N:
                continue
            num = abs(2 * a - 3 * b)
            if num == 0:
                continue
            den = 2 * b
            
            if num * best_err_den < best_err_num * den:
                best_err_num = num
                best_err_den = den
                best_a = a
                best_b = b
    
    if best_err_num == float('inf'):
        return [11, 6]
    
    return [int(best_a), int(best_b)]
