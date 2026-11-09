# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    n = 30
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def enumerate_sums(ws):
        sums = [0]
        for w in ws:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums

    sums1 = enumerate_sums(w1)
    sums2 = enumerate_sums(w2)
    
    sums2.sort()
    
    best_abs = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    for s1 in sums1:
        target = -s1
        idx = bisect_left(sums2, target)
        
        # Check neighbors
        candidates = []
        for i in range(idx - 2, idx + 3):
            if 0 <= i < len(sums2):
                candidates.append(sums2[i])
        
        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_abs:
                best_abs = abs_total
                best_s1 = s1
                best_s2 = s2

    # Precompute suffix sums for pruning in find_coeffs
    def find_coeffs(ws, target_sum):
        length = len(ws)
        suffix_max = [0] * (length + 1)
        for i in range(length - 1, -1, -1):
            suffix_max[i] = suffix_max[i + 1] + ws[i]
            
        coeffs = [0] * length
        
        def dfs(pos, current_sum):
            if pos == length:
                return current_sum == target_sum
            
            w = ws[pos]
            remaining = suffix_max[pos + 1]
            
            # Try -1, 0, 1
            # Order might matter for speed, but any order works for correctness.
            for c in (0, -1, 1): 
                new_sum = current_sum + c * w
                if abs(target_sum - new_sum) <= remaining:
                    coeffs[pos] = c
                    if dfs(pos + 1, new_sum):
                        return True
            
            coeffs[pos] = 0
            return False
        
        if dfs(0, 0):
            return coeffs
        return None

    coeffs1 = find_coeffs(w1, best_s1)
    coeffs2 = find_coeffs(w2, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
    
    return coeffs1 + coeffs2
# EVOLVE-BLOCK-END
