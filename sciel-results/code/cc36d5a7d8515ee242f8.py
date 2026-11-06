"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Strategy: 
    # 1. Generate powerful numbers up to L = 10^8.
    # 2. Use Numpy to compute all pairwise sums a+b (a<=b) efficiently.
    # 3. Count frequencies using np.unique.
    # 4. Identify N with max frequency.
    # 5. Extract pairs for that N.
    
    L = 10**8
    
    # Generate powerful numbers
    # Powerful numbers are of form a^2 * b^3
    powerfuls = set()
    max_b = int(L ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= L:
                powerfuls.add(val)
    
    # Convert to sorted list and numpy array
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    P = np.array(powerfuls_list, dtype=np.int64)
    
    # Check time
    if time.time() > deadline - 30:
        # Fallback if generation was too slow (unlikely)
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Compute all sums a+b for a<=b
    # We create the full matrix of sums, then take the upper triangle
    # This uses M*M memory. For M=8000, 64M elements * 8 bytes = 512MB.
    # This is acceptable.
    
    sums_matrix = P[:, None] + P[None, :]
    
    # Get indices for upper triangle (including diagonal)
    i, j = np.triu_indices(M)
    
    # Extract sums
    sums_upper = sums_matrix[i, j]
    
    # Count occurrences
    # np.unique is C-optimized and fast
    unique_sums, counts = np.unique(sums_upper, return_counts=True)
    
    # Find the sum with the maximum count
    max_idx = np.argmax(counts)
    best_N = int(unique_sums[max_idx])
    max_count = int(counts[max_idx])
    
    # Check time
    if time.time() > deadline - 10:
        # If we are running out of time, we might not be able to verify/extract pairs safely
        # But we need to return valid pairs.
        # If we can't extract, we should return a known valid small pair or try to extract quickly.
        # Extraction is O(M), which is fast.
        pass
        
    # Extract pairs for best_N
    # We need distinct pairs [a, b] with a <= b, a+b=best_N, a,b in powerfuls.
    powerfuls_set = set(powerfuls_list)
    pairs = []
    # Iterate through powerfuls_list (sorted)
    # a <= best_N / 2
    limit_a = best_N // 2
    
    for a in powerfuls_list:
        if a > limit_a:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
            
    # Verify count matches (sanity check)
    # The count from FFT/unique includes all a<=b pairs.
    # Our extraction logic finds exactly those.
    
    # Final check
    if not pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}

# EVOLVE-BLOCK-END
