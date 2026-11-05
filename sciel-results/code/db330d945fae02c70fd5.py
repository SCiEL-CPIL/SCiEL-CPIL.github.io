import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 90s
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000025
    
    # Precompute small part for each number 1..max_val
    small_part = np.arange(max_val, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk < max_val:
            for j in range(pk, max_val, pk):
                small_part[j] //= p
            pk *= p
            
    if time.time() > deadline:
        return 1000

    min_n = 1000
    max_n = 1000000
    
    # Compute max small part in each window of 24
    # Window for n is [n+1, n+24]
    # Indices in small_part: n+1 to n+24
    # n ranges from 1000 to 1000000
    # We need small_part[1001 : 1000025]
    
    max_small = np.zeros(max_n - min_n + 1, dtype=np.int64)
    
    # Initialize with -inf or 0? small_part is >= 1.
    # We want the max of 24 values.
    # We can do this by taking the max of shifted arrays.
    
    # small_part is 1-indexed effectively for our logic?
    # small_part[0] is 0. small_part[1] is 1.
    # n=1000 -> indices 1001..1024
    
    # Let's just loop 24 times and update max
    # This is O(24 * N) which is fast.
    
    # We only care about n in [1000, 1000000]
    # Corresponding indices in small_part:
    # n=1000 -> 1001
    # n=1000000 -> 1000024
    
    # Create a view of the relevant part of small_part
    # We need to handle boundaries carefully.
    
    # Instead of complex views, just do:
    # max_small[i] corresponds to n = min_n + i
    # The window is small_part[min_n + i + 1 : min_n + i + 25]
    
    # We can compute this using a sliding window max algorithm or just simple numpy operations.
    # Given N=10^6, 24 passes is fine.
    
    # Initialize max_small with the first offset
    max_small[:] = small_part[min_n + 1 : min_n + 1 + (max_n - min_n + 1)]
    
    for k in range(1, 24):
        offset = min_n + 1 + k
        # We want to compare small_part[offset : offset + (max_n - min_n + 1)]
        # with current max_small
        np.maximum(max_small, small_part[offset : offset + (max_n - min_n + 1)], out=max_small)
        
    if time.time() > deadline:
        # Return the n with smallest max_small so far
        best_idx = np.argmin(max_small)
        return min_n + best_idx

    # Get the top 10000 candidates with smallest max_small
    # This prioritizes windows where all numbers have small "small parts"
    top_k = 10000
    if len(max_small) > top_k:
        # argsort is O(N log N), but we only need top K.
        # np.argpartition is O(N)
        part_indices = np.argpartition(max_small, top_k)[:top_k]
    else:
        part_indices = np.arange(len(max_small))
        
    best_count = 25
    best_n = min_n + part_indices[0]
    
    # Evaluate these candidates
    for idx in part_indices:
        if time.time() > deadline:
            break
        n = min_n + idx
        # Extract the 24 small parts
        # Indices: n+1 to n+24
        vals = small_part[n+1 : n+25]
        
        # Count distinct values
        # Since vals are small integers (hopefully), we can use a set
        # Or np.unique
        distinct_count = len(np.unique(vals))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                return best_n
                
    return best_n
