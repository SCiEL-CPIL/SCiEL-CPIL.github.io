import numpy as np
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Precompute weights w_i = L // i
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i

    # Split indices into two halves
    n_half = 15
    weights_A = np.array([weights[i] for i in range(1, n_half + 1)], dtype=np.int64)
    weights_B = np.array([weights[i] for i in range(n_half + 1, N + 1)], dtype=np.int64)
    
    num_sums = 3 ** n_half
    
    # Function to generate sums and ms for a set of weights
    def gen_sums_ms(weights_vec):
        # Weights vec is 1D numpy array
        n = len(weights_vec)
        # Create all base-3 representations
        # This is memory heavy if done naively, but 3^15 is 14M.
        # We can generate digits using np.arange and division?
        # m = 0 .. 3^n - 1
        ms = np.arange(num_sums, dtype=np.int64)
        
        # To compute sums, we need to expand ms to digits.
        # This is tricky in numpy without a loop.
        # However, we can use a loop over the digits (15 iterations).
        # This is 15 * 14M operations, which is fast in numpy.
        
        sums = np.zeros(num_sums, dtype=np.int64)
        temp_ms = ms.copy()
        
        for i in range(n):
            digits = temp_ms % 3
            # c = digits - 1
            c = digits - 1
            sums += c * weights_vec[i]
            temp_ms //= 3
            
        return sums, ms

    # Generate for A
    sums_A, ms_A = gen_sums_ms(weights_A)
    
    # Sort A
    order = np.argsort(sums_A)
    sorted_sums_A = sums_A[order]
    sorted_ms_A = ms_A[order]
    
    # Generate for B
    sums_B, ms_B = gen_sums_ms(weights_B)
    
    # Search
    targets = -sums_B
    
    # Find positions
    pos = np.searchsorted(sorted_sums_A, targets)
    
    # Candidate 1: pos - 1
    idx1 = pos - 1
    mask1 = idx1 >= 0
    idx1_valid = idx1[mask1]
    b_idx1 = np.arange(num_sums)[mask1]
    
    sums1 = sorted_sums_A[idx1_valid] + sums_B[b_idx1]
    ms_A_1 = sorted_ms_A[idx1_valid]
    ms_B_1 = ms_B[b_idx1]
    
    # Candidate 2: pos
    idx2 = pos
    mask2 = idx2 < len(sorted_sums_A)
    idx2_valid = idx2[mask2]
    b_idx2 = np.arange(num_sums)[mask2]
    
    sums2 = sorted_sums_A[idx2_valid] + sums_B[b_idx2]
    ms_A_2 = sorted_ms_A[idx2_valid]
    ms_B_2 = ms_B[b_idx2]
    
    # Combine
    all_sums = np.concatenate([sums1, sums2])
    all_ms_A = np.concatenate([ms_A_1, ms_A_2])
    all_ms_B = np.concatenate([ms_B_1, ms_B_2])
    
    # Filter non-zero
    non_zero_mask = all_sums != 0
    if not np.any(non_zero_mask):
        # Should not happen
        return [1] + [0] * (N - 1)
        
    nz_sums = all_sums[non_zero_mask]
    nz_ms_A = all_ms_A[non_zero_mask]
    nz_ms_B = all_ms_B[non_zero_mask]
    
    # Find min abs
    abs_sums = np.abs(nz_sums)
    min_idx = np.argmin(abs_sums)
    best_abs_num = abs_sums[min_idx]
    best_m_A = nz_ms_A[min_idx]
    best_m_B = nz_ms_B[min_idx]
    
    # Reconstruct coefficients
    coeffs = [0] * N
    
    # Decode A
    temp_A = best_m_A
    for i in range(n_half):
        digit = temp_A % 3
        c = digit - 1
        coeffs[i] = c
        temp_A //= 3
        
    # Decode B
    temp_B = best_m_B
    for i in range(n_half):
        digit = temp_B % 3
        c = digit - 1
        coeffs[n_half + i] = c
        temp_B //= 3
        
    return coeffs
