"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce
import numpy as np

def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that the signed sum
        S = sum_{i=1}^{30} c[i-1] / i
    is nonzero and has minimal absolute value.
    
    Uses meet-in-the-middle with NumPy acceleration for the search phase.
    """
    n = 30
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    weights = [L // (i + 1) for i in range(n)]
    half = n // 2
    
    def enumerate_half(start, length):
        sums = [0]
        for i in range(start, start + length):
            w = weights[i]
            new_sums = [0] * (len(sums) * 3)
            idx = 0
            for s in sums:
                new_sums[idx] = s - w
                new_sums[idx + 1] = s
                new_sums[idx + 2] = s + w
                idx += 3
            sums = new_sums
        return sums

    sums1 = enumerate_half(0, half)
    sums2 = enumerate_half(half, n - half)
    
    # Use NumPy for fast sorting and search
    arr2 = np.array(sums2, dtype=np.int64)
    arr2.sort()
    
    best_val = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    for s1 in sums1:
        target = -s1
        idx = np.searchsorted(arr2, target)
        
        # Check idx and idx-1
        if idx > 0:
            s2 = arr2[idx - 1]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_val:
                    best_val = abs_total
                    best_s1 = s1
                    best_s2 = s2
        
        if idx < len(arr2):
            s2 = arr2[idx]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_val:
                    best_val = abs_total
                    best_s1 = s1
                    best_s2 = s2
    
    def find_coeffs(start, length, target_sum):
        coeffs = [0] * length
        
        # Precompute suffix sums for pruning
        suffix_max = [0] * (length + 1)
        for i in range(length - 1, -1, -1):
            suffix_max[i] = suffix_max[i + 1] + weights[start + i]
        
        def dfs(pos, current_sum):
            if pos == length:
                return current_sum == target_sum
            
            w = weights[start + pos]
            rem_max = suffix_max[pos + 1]
            
            for c in (-1, 0, 1):
                coeffs[pos] = c
                new_sum = current_sum + c * w
                gap = target_sum - new_sum
                if -rem_max <= gap <= rem_max:
                    if dfs(pos + 1, new_sum):
                        return True
            
            coeffs[pos] = 0
            return False
        
        if dfs(0, 0):
            return coeffs
        return None

    coeffs1 = find_coeffs(0, half, best_s1)
    coeffs2 = find_coeffs(half, n - half, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
    
    return coeffs1 + coeffs2

# EVOLVE-BLOCK-END
