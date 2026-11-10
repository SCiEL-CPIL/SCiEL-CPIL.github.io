"""Program for small nonzero signed harmonic subsums using Meet-in-the-Middle."""
# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from bisect import bisect_left

def construct():
    n = 30
    
    # Compute L = lcm(1, 2, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w_i = L // i
    # We want to minimize |sum c_i * w_i| where c_i in {-1, 0, 1}
    # and sum c_i * w_i != 0
    
    w = [L // i for i in range(1, n + 1)]
    
    half = n // 2 # 15
    
    # Generate all sums for first half
    # We represent the combination as an integer in base 3.
    # digit 0 -> c = -1
    # digit 1 -> c = 0
    # digit 2 -> c = 1
    
    def generate_sums(weights):
        k = len(weights)
        # num_combos = 3 ** k
        # Instead of pre-allocating and filling, we use iterative concatenation
        current_sums = np.array([0], dtype=np.int64)
        for w_val in weights:
            current_sums = np.concatenate([
                current_sums - w_val, 
                current_sums, 
                current_sums + w_val
            ])
        return current_sums

    # Generate sums for both halves
    sums1 = generate_sums(w[:half])
    sums2 = generate_sums(w[half:])
    
    # Sort sums2 for binary search
    # We need to keep track of original indices to decode coefficients later
    sorted_indices = np.argsort(sums2)
    sums2_sorted = sums2[sorted_indices]
    
    best_abs_val = np.iinfo(np.int64).max
    best_idx1 = -1
    best_idx2 = -1
    
    # Iterate through sums1
    # For each s1, find s2 in sums2_sorted closest to -s1
    # such that s1 + s2 != 0
    
    for i1, s1 in enumerate(sums1):
        target = -s1
        # Find insertion point
        pos = bisect_left(sums2_sorted, target)
        
        # Check neighbors: pos-1 and pos
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(sums2_sorted):
            candidates.append(pos)
        
        for p in candidates:
            if p < 0 or p >= len(sums2_sorted):
                continue
            s2_val = sums2_sorted[p]
            total = s1 + s2_val
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_abs_val:
                best_abs_val = abs_total
                best_idx1 = i1
                best_idx2 = sorted_indices[p]
                
                # Early termination if we find 1? 
                # The minimum non-zero integer sum is 1.
                if best_abs_val == 1:
                    break
        
        if best_abs_val == 1:
            break
            
    if best_idx1 == -1:
        # Fallback (should not happen)
        return [0] * 28 + [1, -1]
        
    # Decode indices to coefficients
    # Index is in base 3.
    # digit 0 -> -1, 1 -> 0, 2 -> 1
    
    def decode_index(idx, k):
        coeffs = []
        for _ in range(k):
            digit = idx % 3
            idx //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs
    
    c1 = decode_index(best_idx1, half)
    c2 = decode_index(best_idx2, n - half)
    
    return c1 + c2

# EVOLVE-BLOCK-END
