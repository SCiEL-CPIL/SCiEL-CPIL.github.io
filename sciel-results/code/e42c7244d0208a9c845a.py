import numpy as np
from math import gcd
from functools import reduce
import time

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    # 1. Calculate LCM of 1..30
    # L = _lcm_list(range(1, 31))
    L = 2329089562800
    
    # 2. Calculate weights W_i = L / i
    weights = np.array([L // (i + 1) for i in range(30)], dtype=np.int64)
    
    # 3. Split into two halves
    n = 30
    n1 = n // 2
    n2 = n - n1
    
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    def generate_sums_and_codes(weights_half):
        """
        Generates all possible sums and their base-3 encoded coefficient vectors.
        Uses pre-allocation for speed.
        """
        n_h = len(weights_half)
        total_size = 3 ** n_h
        
        sums = np.zeros(total_size, dtype=np.int64)
        codes = np.zeros(total_size, dtype=np.int32)
        
        current_size = 1
        power = 1
        
        for j, w in enumerate(weights_half):
            base_sums = sums[:current_size]
            base_codes = codes[:current_size].copy()
            
            # Update Keep (coeff 0, digit 1)
            # sums[0:current_size] remains base_sums (value-wise)
            # But we need to update codes
            codes[0:current_size] = base_codes + 1 * power
            
            # Add (coeff 1, digit 2)
            sums[current_size:2*current_size] = base_sums + w
            codes[current_size:2*current_size] = base_codes + 2 * power
            
            # Sub (coeff -1, digit 0)
            sums[2*current_size:3*current_size] = base_sums - w
            codes[2*current_size:3*current_size] = base_codes # + 0 * power
            
            current_size *= 3
            power *= 3
            
        return sums, codes

    # Generate for first half
    sums1, codes1 = generate_sums_and_codes(w1)
    
    if time.time() - start_time > DEADLINE - 10:
        return [0]*30

    # Generate for second half
    sums2, codes2 = generate_sums_and_codes(w2)
    
    if time.time() - start_time > DEADLINE - 5:
        return [0]*30

    # Sort sums2 for binary search
    order2 = np.argsort(sums2)
    sums2 = sums2[order2]
    codes2 = codes2[order2]
    
    # Vectorized search for closest non-zero sum
    targets = -sums1
    
    # Find insertion points in sums2
    pos = np.searchsorted(sums2, targets)
    
    len2 = len(sums2)
    
    # Candidate 1: index = pos (clamped)
    idx1 = np.clip(pos, 0, len2 - 1)
    s2_1 = sums2[idx1]
    diff1 = np.abs(sums1 + s2_1)
    
    # Candidate 2: index = pos - 1 (clamped)
    idx2 = np.clip(pos - 1, 0, len2 - 1)
    s2_2 = sums2[idx2]
    diff2 = np.abs(sums1 + s2_2)
    
    # Mask zeros
    zero_mask_1 = (diff1 == 0)
    zero_mask_2 = (diff2 == 0)
    
    diff1_valid = np.where(zero_mask_1, np.inf, diff1)
    diff2_valid = np.where(zero_mask_2, np.inf, diff2)
    
    min_diff = np.minimum(diff1_valid, diff2_valid)
    
    if not np.any(np.isfinite(min_diff)):
        return [0]*30

    min_idx = np.argmin(min_diff)
    
    use_idx1 = diff1_valid[min_idx] <= diff2_valid[min_idx]
    
    if use_idx1:
        best_s2_idx = idx1[min_idx]
        best_diff = diff1[min_idx]
    else:
        best_s2_idx = idx2[min_idx]
        best_diff = diff2[min_idx]
        
    if not np.isfinite(best_diff):
        return [0]*30
        
    best_code1 = codes1[min_idx]
    best_code2 = codes2[best_s2_idx]
    
    coeffs = [0] * 30
    
    c1 = int(best_code1)
    for j in range(n1):
        digit = c1 % 3
        c1 //= 3
        if digit == 0:
            coeffs[j] = -1
        elif digit == 1:
            coeffs[j] = 0
        else:
            coeffs[j] = 1
            
    c2 = int(best_code2)
    for j in range(n2):
        digit = c2 % 3
        c2 //= 3
        if digit == 0:
            coeffs[n1 + j] = -1
        elif digit == 1:
            coeffs[n1 + j] = 0
        else:
            coeffs[n1 + j] = 1
            
    return coeffs
