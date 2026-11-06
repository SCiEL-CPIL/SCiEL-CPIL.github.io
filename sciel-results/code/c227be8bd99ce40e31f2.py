import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for 150s limit
    
    # Choose L based on memory/time tradeoff.
    # L=5*10^8 -> M~17k -> Sums~150M -> 1.2GB. Sort takes ~5s.
    # L=10^9 -> M~100k -> Sums~5B -> 40GB. Too big.
    # L=2*10^8 -> M~12.6k -> Sums~80M -> 640MB.
    
    L = 5 * 10**8
    
    # 1. Generate powerful numbers up to L
    # Powerful numbers are of the form a^2 * b^3.
    # We iterate b, then a.
    powerful_set = set()
    max_b = int(L ** (1/3)) + 1
    
    for b in range(1, max_b):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        if max_a < 1:
            continue
        # Vectorized generation for this b
        # a_vals = np.arange(1, max_a + 1)
        # p_vals = (a_vals * a_vals) * b3
        # This creates an array of size max_a.
        # For b=1, max_a is isqrt(5e8) ~ 22360.
        # Total operations is sum of max_a over b.
        # This is roughly sum_{b=1}^{L^{1/3}} L^{1/2} / b^{3/2} ~ L^{1/2} * constant.
        # L=5e8 -> sqrt(L) ~ 22360.
        # Total numbers generated is small.
        
        a_vals = np.arange(1, max_a + 1)
        p_vals = (a_vals * a_vals) * b3
        # Filter if any > L (shouldn't happen, but safe)
        # Add to set
        # Converting to list and updating set is O(N) per b.
        # Total elements ~ 10k-20k. This is fast.
        powerful_set.update(p_vals.tolist())
        
    if time.time() - start_time > time_limit * 0.1:
        # If generation is unexpectedly slow, fallback to smaller L?
        # Unlikely.
        pass

    A = np.array(sorted(powerful_set), dtype=np.int64)
    M = len(A)
    
    # 2. Compute pairwise sums (i <= j)
    # Total number of sums is M*(M+1)/2
    n_sums = M * (M + 1) // 2
    
    # Check if we can allocate this array
    # 8 bytes per element
    mem_required = n_sums * 8
    # Assume 4GB is safe limit for this array + overhead
    if mem_required > 4 * 1024**3:
        # Fallback to smaller L if M is too large
        # This shouldn't happen with L=5e8 (M~17k, n_sums~150M, 1.2GB)
        L = 10**8
        powerful_set = set()
        max_b = int(L ** (1/3)) + 1
        for b in range(1, max_b):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            if max_a < 1:
                continue
            a_vals = np.arange(1, max_a + 1)
            p_vals = (a_vals * a_vals) * b3
            powerful_set.update(p_vals.tolist())
        A = np.array(sorted(powerful_set), dtype=np.int64)
        M = len(A)
        n_sums = M * (M + 1) // 2

    sums = np.empty(n_sums, dtype=np.int64)
    
    # Fill the sums array
    # We fill it row by row.
    # Row i contains A[i] + A[i], A[i] + A[i+1], ..., A[i] + A[M-1]
    # Length of row i is M - i.
    offset = 0
    for i in range(M):
        row_len = M - i
        # sums[offset : offset + row_len] = A[i] + A[i:]
        sums[offset : offset + row_len] = A[i] + A[i:]
        offset += row_len
        
    if time.time() - start_time > time_limit * 0.3:
        # If filling is slow, we might be in trouble, but it's vectorized.
        pass

    # 3. Sort and count
    # Sorting 150M elements.
    sums.sort()
    
    # Count consecutive identical values
    # We can use np.diff to find where values change.
    # changes[i] = 1 if sums[i] != sums[i-1] else 0
    # This is O(N).
    changes = np.diff(sums) != 0
    # The indices where change happens are the ends of groups.
    # Let's find the start indices of groups.
    # Group 0 starts at 0.
    # Next group starts at index of first change + 1.
    
    # More efficient:
    # unique_sums, counts = np.unique(sums, return_counts=True)
    # np.unique sorts anyway, so we are double sorting?
    # No, if sums is already sorted, np.unique is faster?
    # Actually, np.unique on a sorted array is O(N).
    # Let's use np.unique.
    
    unique_sums, counts = np.unique(sums, return_counts=True)
    
    # Find the max count
    max_idx = np.argmax(counts)
    best_N = unique_sums[max_idx]
    max_pairs = counts[max_idx]
    
    # Verify N <= 10^18
    if best_N > 10**18:
        # Find next best
        # This is unlikely for L=5e8 (max sum 1e9)
        pass
        
    # Extract pairs
    # We need to return the list of pairs.
    # max_pairs can be large.
    # We need to find all a in A such that N-a in A and a <= N-a.
    
    A_set = set(A.tolist())
    pairs = []
    # Iterate through A
    # Optimization: Only iterate a <= N/2
    half_N = best_N // 2
    # Find indices in A where A[i] <= half_N
    # Since A is sorted, we can use searchsorted
    idx_limit = np.searchsorted(A, half_N)
    
    for i in range(idx_limit):
        a = A[i]
        b = best_N - a
        if b in A_set:
            pairs.append([int(a), int(b)])
            
    # Check time
    if time.time() - start_time > time_limit:
        # Return what we have
        pass

    return {'N': int(best_N), 'pairs': pairs}
