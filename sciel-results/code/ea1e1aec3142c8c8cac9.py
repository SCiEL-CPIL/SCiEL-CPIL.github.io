import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    # Deadline guard: 110 seconds
    deadline = start_time + 110.0
    
    # Strategy: Use FFT to count sums of powerful numbers.
    # This allows handling larger M than pure Python O(K^2) loops.
    # We aim for M such that the FFT array size (2*M) fits in memory.
    # 10^7 elements * 8 bytes/complex = 80MB per array. 
    # We need a few arrays. 200-300MB is safe.
    # Let's try M = 10^7.
    M = 10_000_000
    
    # 1. Generate powerful numbers up to M
    powerful_set = set()
    max_v = int(M ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M:
            break
        max_u = isqrt(M // v_cubed)
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            powerful_set.add(p)
            
    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    
    # 2. Prepare for FFT
    # We need an array of size at least 2*M + 1 to avoid circular convolution artifacts.
    # Next power of 2 for FFT efficiency.
    size = 1
    while size < 2 * M + 1:
        size *= 2
        
    # Check time
    if time.time() > deadline - 30:
         # Fallback to smaller M or simple loop if generation took too long
         # But generation is fast. FFT is the heavy part.
         pass

    # Create indicator array
    # Using float64 for FFT
    A = np.zeros(size, dtype=np.float64)
    for p in powerful_list:
        A[p] = 1.0
        
    # 3. FFT Convolution
    # FFT(A) * FFT(A) -> IFFT -> Counts
    # We can use rfft/irfft for real inputs
    F = np.fft.rfft(A)
    F_sq = F * F
    # Inverse FFT
    # The result will be real, but may have small imaginary parts due to floating point errors
    counts_float = np.fft.irfft(F_sq, n=size)
    
    # The counts are in counts_float[0:size]
    # We only care about N up to 2*M
    # Round to nearest integer
    counts_int = np.round(counts_float[:2*M + 1]).astype(np.int64)
    
    # 4. Find candidates
    # The convolution counts ordered pairs (a,b).
    # We want distinct pairs a <= b.
    # Approximate score is counts_int[N] // 2.
    # Let's find the top 100 indices with highest counts_int.
    
    # Get top 100 indices
    # argsort is O(N log N). N = 2*10^7. This is fast in numpy.
    # But we can just take argmax of a subset?
    # Let's just find the global max first.
    max_val = np.max(counts_int)
    if max_val == 0:
        # Fallback if something went wrong
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find all indices with count close to max?
    # Or just take top 50 to verify exact counts.
    # We can use argpartition to get top K unsorted.
    top_k = 50
    top_indices = np.argpartition(counts_int, -top_k)[-top_k:]
    
    # 5. Verify exact counts for these candidates
    best_N = 0
    best_exact_count = 0
    best_pairs = []
    
    powerful_set_local = powerful_set # For O(1) lookup
    
    for idx in top_indices:
        N = int(idx)
        # Calculate exact number of pairs a <= b, a+b=N, a,b powerful
        count = 0
        pairs = []
        # Iterate through powerful numbers <= N/2
        # We can binary search in powerful_list
        # Or just iterate all if K is small enough?
        # K ~ 4500. Iterating all K for 50 candidates is 225,000 checks. Very fast.
        
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set_local:
                count += 1
                pairs.append([a, b])
                
        if count > best_exact_count:
            best_exact_count = count
            best_N = N
            best_pairs = pairs
            
    # Check time
    if time.time() > deadline - 5:
        # Return what we have
        pass

    return {'N': best_N, 'pairs': best_pairs}
