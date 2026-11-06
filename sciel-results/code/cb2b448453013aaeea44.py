"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import gcd

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    N = 30
    half = 15
    
    # Denominators are 1, 2, ..., 30
    # Use global LCM as common denominator
    # LCM(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    # L = 465817912560
    
    # For each denominator d, the coefficient c/d = c * (L/d) / L
    # So we need integer weights w[i] = L // (i+1) for i=0..29
    weights = np.array([L // (i + 1) for i in range(N)], dtype=np.int64)
    
    # Split into two halves
    w1 = weights[:half]  # denominators 1..15
    w2 = weights[half:]  # denominators 16..30
    
    def gen_sums_and_codes(weights_arr, time_limit):
        """Generate all possible sums AND their coefficient codes for given weights."""
        n = len(weights_arr)
        # Use iterative approach: maintain list of (sum, code) pairs
        # code is a base-3 number representing the coefficients
        
        # Start with [(0, 0)]
        sums = [0]
        codes = [0]
        
        for i, w in enumerate(weights_arr):
            new_sums = []
            new_codes = []
            # For each existing sum, create three new sums: sum-w, sum, sum+w
            # Code digit: 0 -> -1, 1 -> 0, 2 -> +1
            for s, c in zip(sums, codes):
                new_sums.append(s - w)
                new_codes.append(c)  # digit 0 means -1
                new_sums.append(s)
                new_codes.append(c + 3**i)  # digit 1 means 0
                new_sums.append(s + w)
                new_codes.append(c + 2 * 3**i)  # digit 2 means +1
            
            sums = new_sums
            codes = new_codes
            
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
        
        return np.array(sums, dtype=np.int64), np.array(codes, dtype=np.int64)
    
    # Generate sums and codes for half 1
    sums1, codes1 = gen_sums_and_codes(w1, deadline - 20.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        # Fallback: return a simple valid solution
        return [0]*28 + [1, -1]
    
    # Generate sums and codes for half 2
    sums2, codes2 = gen_sums_and_codes(w2, deadline - 5.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 3.0:
        return [0]*28 + [1, -1]
    
    # Sort sums1 for binary search, keep track of original indices
    order1 = np.argsort(sums1)
    sums1_sorted = sums1[order1]
    codes1_sorted = codes1[order1]
    
    # For each value in sums2, find the value in sums1 that makes sum closest to 0
    # We want a + b ≈ 0, i.e., a ≈ -b
    # For each b in sums2, find a in sums1_sorted closest to -b
    
    neg_sums2 = -sums2
    # Find where each -b would be inserted in sums1_sorted
    idx = np.searchsorted(sums1_sorted, neg_sums2)
    
    # Check candidates: idx-1 and idx
    best_abs = None
    best_a = None
    best_b = None
    best_code1 = None
    best_code2 = None
    
    # Process in chunks to manage memory
    chunk_size = 50000
    for start in range(0, len(sums2), chunk_size):
        end = min(start + chunk_size, len(sums2))
        neg_chunk = neg_sums2[start:end]
        idx_chunk = idx[start:end]
        
        # Candidate indices: idx-1 and idx
        # Clamp to valid range
        idx_left = np.clip(idx_chunk - 1, 0, len(sums1_sorted) - 1)
        idx_right = np.clip(idx_chunk, 0, len(sums1_sorted) - 1)
        
        a_left = sums1_sorted[idx_left]
        a_right = sums1_sorted[idx_right]
        
        b_chunk = sums2[start:end]
        
        # Compute sums
        sum_left = a_left + b_chunk
        sum_right = a_right + b_chunk
        
        abs_left = np.abs(sum_left)
        abs_right = np.abs(sum_right)
        
        # Exclude zero sums
        abs_left[sum_left == 0] = np.iinfo(np.int64).max
        abs_right[sum_right == 0] = np.iinfo(np.int64).max
        
        # Find best in this chunk
        min_left = np.min(abs_left)
        min_right = np.min(abs_right)
        
        if min_left < min_right:
            local_best_idx = np.argmin(abs_left)
            local_best_abs = min_left
            local_best_a = a_left[local_best_idx]
            local_best_b = b_chunk[local_best_idx]
            local_best_code1 = codes1_sorted[idx_left[local_best_idx]]
            local_best_code2 = codes2[start + local_best_idx]
        else:
            local_best_idx = np.argmin(abs_right)
            local_best_abs = min_right
            local_best_a = a_right[local_best_idx]
            local_best_b = b_chunk[local_best_idx]
            local_best_code1 = codes1_sorted[idx_right[local_best_idx]]
            local_best_code2 = codes2[start + local_best_idx]
        
        if best_abs is None or local_best_abs < best_abs:
            best_abs = local_best_abs
            best_a = local_best_a
            best_b = local_best_b
            best_code1 = local_best_code1
            best_code2 = local_best_code2
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            break
    
    if best_abs is None or best_abs == np.iinfo(np.int64).max:
        # No valid solution found, fallback
        return [0]*28 + [1, -1]
    
    # Decode the codes back to coefficients
    def decode_code(code, n):
        """Decode a base-3 code into coefficients in {-1, 0, 1}."""
        coeffs = []
        for i in range(n):
            digit = code % 3
            code //= 3
            # digit 0 -> -1, digit 1 -> 0, digit 2 -> +1
            coeffs.append(digit - 1)
        return coeffs
    
    coeffs1 = decode_code(best_code1, half)
    coeffs2 = decode_code(best_code2, half)
    
    # Combine
    result = coeffs1 + coeffs2
    
    # Verify
    total = sum(result[i] / (i + 1) for i in range(N))
    if total == 0:
        # Shouldn't happen, but fallback
        return [0]*28 + [1, -1]
    
    return result

# EVOLVE-BLOCK-END
