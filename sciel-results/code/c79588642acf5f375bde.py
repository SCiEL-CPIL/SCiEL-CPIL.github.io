"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # 2. Calculate weights W_i = L // i
    weights = [L // i for i in range(1, 31)]
    
    # 3. Split weights into two halves
    half1_weights = weights[:15]
    half2_weights = weights[15:]
    
    def generate_sums(ws):
        """Generate all possible signed sums for the given weights."""
        sums = np.array([0], dtype=np.int64)
        for w in ws:
            # Create new array: [x-w, x, x+w for x in sums]
            # This triples the size
            sums = np.concatenate((sums - w, sums, sums + w))
        return sums
    
    # Generate sums for both halves
    sums1 = generate_sums(half1_weights)
    sums2 = generate_sums(half2_weights)
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback to seed if generation took too long (unlikely for 3^15)
        return [0] * 28 + [1, -1]

    # 4. Sort sums2 to allow binary search
    sorted_sums2 = np.sort(sums2)
    
    # 5. Find minimum non-zero sum
    # For each s1 in sums1, we want s2 such that s1 + s2 is close to 0 but not 0.
    # We search for -s1 in sorted_sums2.
    
    # Vectorized search
    # idx is the index where -s1 would be inserted to maintain order
    # If -s1 exists, idx points to the first occurrence (using side='left')
    idx = np.searchsorted(sorted_sums2, -sums1, side='left')
    
    # We need to check the element at idx and idx-1 (if valid)
    # Also, if -s1 is not present, the closest values are at idx and idx-1.
    
    # Create arrays for candidate s2 values
    # Clip indices to valid range [0, len(sorted_sums2)-1]
    idx_clipped = np.clip(idx, 0, len(sorted_sums2) - 1)
    idx_minus_1_clipped = np.clip(idx - 1, 0, len(sorted_sums2) - 1)
    
    cand_s2_1 = sorted_sums2[idx_clipped]
    cand_s2_2 = sorted_sums2[idx_minus_1_clipped]
    
    total_1 = sums1 + cand_s2_1
    total_2 = sums1 + cand_s2_2
    
    # We want min |total| such that total != 0
    # Mask out zeros
    mask_1 = total_1 != 0
    mask_2 = total_2 != 0
    
    # If no valid candidates in mask_1, set abs to inf
    abs_total_1 = np.where(mask_1, np.abs(total_1), np.inf)
    abs_total_2 = np.where(mask_2, np.abs(total_2), np.inf)
    
    # Find global minimum
    min_val_1 = np.min(abs_total_1)
    min_val_2 = np.min(abs_total_2)
    
    if min_val_1 <= min_val_2:
        min_abs_val = min_val_1
        # Find the index in sums1 that achieved this
        # We need to handle ties: pick the first one
        # Also need to ensure we pick a valid non-zero one
        # np.argmin returns the first occurrence
        # But if min_val_1 is inf, we have a problem
        if min_abs_val == np.inf:
            # Should not happen, but fallback
            return [0] * 28 + [1, -1]
        
        k1_idx = np.argmin(abs_total_1)
        s1_val = sums1[k1_idx]
        s2_val = total_1[k1_idx] - s1_val # This is cand_s2_1[k1_idx]
        
        # Find index of s1_val in sums1
        # Note: sums1 might have duplicates. We just need one.
        # np.where(sums1 == s1_val)[0][0]
        idx_in_sums1 = np.where(sums1 == s1_val)[0][0]
        
        # Find index of s2_val in sums2 (original unsorted)
        # Note: s2_val is a value from sorted_sums2, but we need the index in the original sums2 array
        # to decode the coefficients.
        # Wait, my decoding logic relies on the index in the *generated* array.
        # The generated array `sums2` is in a specific order.
        # `sorted_sums2` is a permuted version.
        # I need to find an index `j` in `sums2` such that `sums2[j] == s2_val`.
        idx_in_sums2 = np.where(sums2 == s2_val)[0][0]
        
    else:
        min_abs_val = min_val_2
        if min_abs_val == np.inf:
            return [0] * 28 + [1, -1]
            
        k1_idx = np.argmin(abs_total_2)
        s1_val = sums1[k1_idx]
        s2_val = total_2[k1_idx] - s1_val
        
        idx_in_sums1 = np.where(sums1 == s1_val)[0][0]
        idx_in_sums2 = np.where(sums2 == s2_val)[0][0]

    # 6. Decode indices to coefficients
    def decode_index(k, weights_half):
        """
        Decode an index k in the generated sums array back to coefficients.
        The generation process was:
        s = [0]
        for w in weights:
            s = [x-w, x, x+w for x in s]
        
        This corresponds to a base-3 representation where:
        digit 0 -> -1
        digit 1 -> 0
        digit 2 -> +1
        
        The most significant digit corresponds to the last weight processed.
        So we process weights in reverse order.
        """
        coeffs = []
        for w in reversed(weights_half):
            # The index k determines the digit
            # k % 3:
            # 0: came from x-w (coeff -1)
            # 1: came from x (coeff 0)
            # 2: came from x+w (coeff 1)
            digit = k % 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
            k //= 3
        # The coeffs are in reverse order of weights
        coeffs.reverse()
        return coeffs

    coeffs1 = decode_index(idx_in_sums1, half1_weights)
    coeffs2 = decode_index(idx_in_sums2, half2_weights)
    
    result = coeffs1 + coeffs2
    
    # Verify result
    # Sum should be non-zero and match min_abs_val * L?
    # Actually, the sum in fractions is min_abs_val / L.
    # We just need to return the coefficients.
    
    return result

# EVOLVE-BLOCK-END
