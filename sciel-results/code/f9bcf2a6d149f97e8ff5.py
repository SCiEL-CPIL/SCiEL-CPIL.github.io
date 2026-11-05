"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def construct():
    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    # w[i-1] = L / i
    w = np.array([L // i for i in range(1, 31)], dtype=np.int64)
    
    def generate_sums(weights):
        n = len(weights)
        current_sums = np.array([0], dtype=np.int64)
        current_ids = np.array([0], dtype=np.int64)
        
        for i in range(n):
            w_i = weights[i]
            len_curr = len(current_sums)
            
            # Create codes array: 0 for -1, 1 for 0, 2 for 1
            # Order in concatenation: [S - w, S, S + w]
            # So codes are [0, 0, ..., 1, 1, ..., 2, 2, ...]
            codes = np.concatenate([
                np.zeros(len_curr, dtype=np.int64),
                np.ones(len_curr, dtype=np.int64),
                np.full(len_curr, 2, dtype=np.int64)
            ])
            
            new_sums = np.concatenate([
                current_sums - w_i,
                current_sums,
                current_sums + w_i
            ])
            
            new_ids = current_ids * 3 + codes
            
            current_sums = new_sums
            current_ids = new_ids
            
        return current_sums, current_ids

    # Split weights into two halves
    w1 = w[:15]
    w2 = w[15:]
    
    # Generate sums for first half
    sums1, ids1 = generate_sums(w1)
    
    # Sort sums1 and keep ids aligned
    sort_idx = np.argsort(sums1)
    sums1_sorted = sums1[sort_idx]
    ids1_sorted = ids1[sort_idx]
    
    # Generate sums for second half
    sums2, ids2 = generate_sums(w2)
    
    # Vectorized search for minimal non-zero absolute sum
    targets = -sums2
    pos = np.searchsorted(sums1_sorted, targets)
    
    # Clip positions to valid range [0, len(sums1_sorted)-1]
    pos_clipped = np.clip(pos, 0, len(sums1_sorted) - 1)
    
    # Candidate 1: sums1_sorted[pos_clipped]
    cand1_sums = sums1_sorted[pos_clipped]
    cand1_ids = ids1_sorted[pos_clipped]
    total1 = cand1_sums + sums2
    abs_total1 = np.abs(total1)
    # Mask out zero sums
    abs_total1[total1 == 0] = np.iinfo(np.int64).max
    
    # Candidate 2: sums1_sorted[pos_clipped - 1]
    pos_minus_1 = np.clip(pos_clipped - 1, 0, len(sums1_sorted) - 1)
    cand2_sums = sums1_sorted[pos_minus_1]
    cand2_ids = ids1_sorted[pos_minus_1]
    total2 = cand2_sums + sums2
    abs_total2 = np.abs(total2)
    # Mask out zero sums
    abs_total2[total2 == 0] = np.iinfo(np.int64).max
    
    # Find global minimum
    min1 = np.min(abs_total1)
    min2 = np.min(abs_total2)
    
    if min1 < min2:
        best_idx = np.argmin(abs_total1)
        best_id1 = cand1_ids[best_idx]
        best_id2 = ids2[best_idx]
        best_val = total1[best_idx]
    else:
        best_idx = np.argmin(abs_total2)
        best_id1 = cand2_ids[best_idx]
        best_id2 = ids2[best_idx]
        best_val = total2[best_idx]
        
    # Decode IDs to coefficients
    def decode_id(id_val, length):
        coeffs = [0] * length
        for i in range(length - 1, -1, -1):
            digit = id_val % 3
            id_val //= 3
            if digit == 0:
                coeffs[i] = -1
            elif digit == 1:
                coeffs[i] = 0
            else:
                coeffs[i] = 1
        return coeffs
    
    coeffs1 = decode_id(int(best_id1), 15)
    coeffs2 = decode_id(int(best_id2), 15)
    
    coeffs = coeffs1 + coeffs2
    
    return coeffs
# EVOLVE-BLOCK-END
