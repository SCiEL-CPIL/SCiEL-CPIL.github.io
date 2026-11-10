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
    DEADLINE = 80.0  # Safety margin (90s limit, 10s buffer)
    
    # 1. Calculate LCM of 1..30
    L = _lcm_list(range(1, 31))
    
    # 2. Calculate weights W_i = L / i
    weights = np.array([L // (i + 1) for i in range(30)], dtype=np.int64)
    
    # 3. Split into two halves
    n = 30
    n1 = n // 2
    n2 = n - n1
    
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    def generate_sums_direct(weights_half):
        """
        Directly enumerate all 3^k combinations using base-3 digits.
        Returns array of sums (int64) and codes (int32).
        """
        k = len(weights_half)
        total = 3 ** k
        sums = np.empty(total, dtype=np.int64)
        codes = np.empty(total, dtype=np.int32)
        
        # For each index, decode the base-3 representation
        # code = sum of digit_j * 3^j where digit_j in {0,1,2}
        # maps to coeff: 0->-1, 1->0, 2->1
        
        for idx in range(total):
            code = idx
            s = 0
            tmp = code
            for j in range(k):
                digit = tmp % 3
                tmp //= 3
                if digit == 0:
                    s -= weights_half[j]
                elif digit == 2:
                    s += weights_half[j]
                # digit == 1: coeff = 0, no contribution
                code = idx  # we store idx as code
            sums[idx] = s
            codes[idx] = code
        
        return sums, codes
    
    # Alternative: vectorized generation using broadcasting
    def generate_sums_vectorized(weights_half):
        """
        Vectorized generation of all 3^k sums.
        """
        k = len(weights_half)
        total = 3 ** k
        
        # Create all base-3 numbers from 0 to 3^k - 1
        codes = np.arange(total, dtype=np.int32)
        
        # For each position j, extract digit j
        sums = np.zeros(total, dtype=np.int64)
        
        for j in range(k):
            power = 3 ** j
            # Extract digit j: (code // power) % 3
            digits = (codes // power) % 3
            # Map: 0 -> -1, 1 -> 0, 2 -> +1
            # coeff = digits - 1
            coeffs = digits.astype(np.int64) - 1
            sums += coeffs * weights_half[j]
        
        return sums, codes
    
    # Check time before generating
    if time.time() - start_time > DEADLINE - 40:
        # Fallback: return a simple valid solution
        coeffs = [0] * 30
        coeffs[0] = 1  # sum = 1, nonzero
        return coeffs
    
    # Generate for first half
    t0 = time.time()
    sums1, codes1 = generate_sums_vectorized(w1)
    t1 = time.time()
    
    if time.time() - start_time > DEADLINE - 30:
        coeffs = [0] * 30
        coeffs[0] = 1
        return coeffs
    
    # Generate for second half
    sums2, codes2 = generate_sums_vectorized(w2)
    t2 = time.time()
    
    if time.time() - start_time > DEADLINE - 20:
        coeffs = [0] * 30
        coeffs[0] = 1
        return coeffs
    
    # Sort sums2 for binary search
    order2 = np.argsort(sums2)
    sums2_sorted = sums2[order2]
    codes2_sorted = codes2[order2]
    
    # Vectorized search: for each s1, find closest s2 to -s1
    # Use searchsorted to find insertion points
    targets = -sums1
    pos = np.searchsorted(sums2_sorted, targets)
    
    # Clamp positions
    pos_clamped = np.clip(pos, 0, len(sums2_sorted) - 1)
    
    # Get candidate s2 values at pos and pos-1
    s2_at_pos = sums2_sorted[pos_clamped]
    s2_at_prev = sums2_sorted[np.clip(pos_clamped - 1, 0, len(sums2_sorted) - 1)]
    
    # Compute total sums for both candidates
    total_pos = sums1 + s2_at_pos
    total_prev = sums1 + s2_at_prev
    
    # Filter out zeros
    valid_pos = total_pos != 0
    valid_prev = total_prev != 0
    
    # Find minimum absolute value
    abs_pos = np.where(valid_pos, np.abs(total_pos), np.iinfo(np.int64).max)
    abs_prev = np.where(valid_prev, np.abs(total_prev), np.iinfo(np.int64).max)
    
    # Combine
    best_idx = np.argmin(np.minimum(abs_pos, abs_prev))
    
    # Determine which candidate is best
    if abs_pos[best_idx] <= abs_prev[best_idx]:
        best_code1 = codes1[best_idx]
        best_code2 = codes2_sorted[pos_clamped[best_idx]]
    else:
        best_code1 = codes1[best_idx]
        best_code2 = codes2_sorted[np.clip(pos_clamped[best_idx] - 1, 0, len(sums2_sorted) - 1)]
    
    # Decode coefficients
    coeffs = [0] * 30
    
    # Decode code1 (first n1 coeffs)
    c1 = best_code1
    for j in range(n1):
        digit = c1 % 3
        c1 //= 3
        if digit == 0:
            coeffs[j] = -1
        elif digit == 1:
            coeffs[j] = 0
        else:
            coeffs[j] = 1
            
    # Decode code2 (next n2 coeffs)
    c2 = best_code2
    for j in range(n2):
        digit = c2 % 3
        c2 //= 3
        if digit == 0:
            coeffs[n1 + j] = -1
        elif digit == 1:
            coeffs[n1 + j] = 0
        else:
            coeffs[n1 + j] = 1
            
    # Verify the result is valid
    L_check = _lcm_list(range(1, 31))
    total = 0
    for i in range(30):
        total += coeffs[i] * (L_check // (i + 1))
    
    if total == 0:
        # Fallback
        coeffs[0] = 1
    
    return coeffs
