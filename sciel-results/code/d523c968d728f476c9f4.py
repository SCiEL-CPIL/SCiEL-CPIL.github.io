import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate powerful numbers up to L
    L = 500_000_000
    
    # 1. Sieve for square-free numbers up to L^(1/3)
    max_b = int(L ** (1.0 / 3.0)) + 2
    is_sqfree = np.ones(max_b + 1, dtype=bool)
    is_sqfree[0] = False
    for p in range(2, int(max_b**0.5) + 1):
        p2 = p * p
        for j in range(p2, max_b + 1, p2):
            is_sqfree[j] = False
            
    sqfree = [b for b in range(1, max_b + 1) if is_sqfree[b]]
    
    # 2. Generate powerful numbers
    powerful_set = set()
    for b in sqfree:
        b3 = b ** 3
        if b3 > L:
            break
        max_c = isqrt(L // b3)
        c_vals = np.arange(1, max_c + 1, dtype=np.int64)
        vals = c_vals * c_vals * b3
        vals = vals[vals <= L]
        if len(vals) > 0:
            powerful_set.update(vals.tolist())
            
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}

    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    P = np.array(powerful_list, dtype=np.int64)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Calculate total number of pairs
    total_pairs = K * (K + 1) // 2
    
    # Allocate array for all sums
    # If memory is an issue, we can fall back to chunked bincount accumulation
    # But 2.4e8 int64s is ~2GB.
    try:
        all_sums = np.empty(total_pairs, dtype=np.int64)
        idx = 0
        # Fill all_sums with P[i] + P[j] for i <= j
        # Vectorized chunking
        chunk_size = 1000
        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            chunk = P[start:end]
            # For each row i in chunk, we need P[i] + P[i:]
            # This is tricky to vectorize directly into a flat array without masks.
            # Let's just do a loop for i, but vectorize the addition.
            # This is O(K^2) in Python loop overhead if we do K iterations.
            # K=22000. 22000 iterations is fine.
            for i in range(start, end):
                # Sums for row i are P[i] + P[i:]
                # Length is K - i
                length = K - i
                all_sums[idx:idx+length] = P[i] + P[i:]
                idx += length
                
        if time.time() > deadline - 30:
            # If filling takes too long, just bincount what we have?
            # No, we need all.
            pass
            
        counts = np.bincount(all_sums, minlength=2 * L + 1)
        
    except MemoryError:
        # Fallback to chunked accumulation
        counts = np.zeros(2 * L + 1, dtype=np.int32)
        for i in range(K):
            sums = P[i] + P[i:]
            counts += np.bincount(sums, minlength=2 * L + 1).astype(np.int32)
            if time.time() > deadline - 10:
                break

    best_N = int(np.argmax(counts))
    
    P_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
