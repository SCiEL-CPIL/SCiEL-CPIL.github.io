import time
import numpy as np
from math import gcd, lcm
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Integer weights: w_i = L // (i+1) for coefficient c_i multiplying 1/(i+1)
    # Sum = sum(c_i * L/(i+1)) / L
    # Minimize |sum(c_i * w_i)| where w_i = L//(i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    # Split into two halves of 15
    half1 = weights[:15]
    half2 = weights[15:]
    
    def gen_sums(w):
        """Generate all 3^len(w) possible sums"""
        sums = [0]
        for wi in w:
            new_sums = []
            for s in sums:
                new_sums.append(s - wi)
                new_sums.append(s)
                new_sums.append(s + wi)
            sums = new_sums
        return sums

    # Generate sums for both halves
    # This might be slow in pure python, but 14M items is borderline.
    # Let's try to optimize generation.
    
    sums1 = gen_sums(half1)
    sums1.sort()
    
    sums2 = gen_sums(half2)
    
    best_abs = float('inf')
    best_K = 0
    best_s1 = 0
    best_s2 = 0
    found_optimal = False
    
    # Search for best pair
    for s2 in sums2:
        # Target s1 such that s1 + s2 is close to 0
        target = -s2
        # Binary search in sums1
        idx = bisect.bisect_left(sums1, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(sums1):
            candidates.append(sums1[idx])
        if idx > 0:
            candidates.append(sums1[idx-1])
            
        for s1 in candidates:
            K = s1 + s2
            if K == 0:
                continue
            abs_K = abs(K)
            if abs_K < best_abs:
                best_abs = abs_K
                best_K = K
                best_s1 = s1
                best_s2 = s2
                
                # Check if this K yields numerator 1
                # N = K / gcd(K, L)
                # If K divides L, then gcd(K, L) = K, so N = 1
                if L % abs_K == 0:
                    found_optimal = True
                    break
        
        if found_optimal:
            break
            
        if time.time() > deadline - 5.0:
            break

    if best_K == 0:
        # Fallback
        fallback = [0] * 28 + [1, -1]
        return fallback

    # Reconstruct coefficients
    def find_coeffs(weights_list, target):
        n_w = len(weights_list)
        # Precompute suffix sums for pruning
        suffix_max = [0] * (n_w + 1)
        for k in range(n_w - 1, -1, -1):
            suffix_max[k] = suffix_max[k + 1] + abs(weights_list[k])
        
        def dfs(idx, cur_sum, coeffs):
            if idx == n_w:
                if cur_sum == target:
                    return list(coeffs)
                return None
            
            # Prune
            if abs(target - cur_sum) > suffix_max[idx]:
                return None
            
            w = weights_list[idx]
            
            # Try -1
            coeffs.append(-1)
            res = dfs(idx + 1, cur_sum - w, coeffs)
            if res is not None:
                return res
            coeffs.pop()
            
            # Try 0
            coeffs.append(0)
            res = dfs(idx + 1, cur_sum, coeffs)
            if res is not None:
                return res
            coeffs.pop()
            
            # Try 1
            coeffs.append(1)
            res = dfs(idx + 1, cur_sum + w, coeffs)
            if res is not None:
                return res
            coeffs.pop()
            
            return None
        
        return dfs(0, 0, [])

    c1 = find_coeffs(half1, int(best_s1))
    c2 = find_coeffs(half2, int(best_s2))
    
    if c1 is None or c2 is None:
        fallback = [0] * 28 + [1, -1]
        return fallback
    
    result = c1 + c2
    
    # Verify
    total_int = sum(c * w for c, w in zip(result, weights))
    if total_int == 0:
        fallback = [0] * 28 + [1, -1]
        return fallback
        
    return result
