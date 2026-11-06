import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Set limit
    # L = 10**12 gives ~10k powerful numbers, ~50M sums, ~400MB memory
    # L = 5*10**12 gives ~17k powerful numbers, ~150M sums, ~1.2GB memory
    # Let's try 2*10**12 as a balance
    L = 2 * 10**12
    
    powerfuls = set()
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    P = np.array(sorted(powerfuls), dtype=np.int64)
    M = len(P)
    
    if time.time() > deadline - 40:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Generate sums a + b where a <= b
    # We do this in chunks to avoid holding too many temporary arrays at once if needed,
    # but for M=12k, a list of arrays is fine.
    sums_chunks = []
    for i in range(M):
        ai = P[i]
        # P[i:] is a view, ai + P[i:] is a new array
        sums_chunks.append(ai + P[i:])
        
    if time.time() > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}

    all_sums = np.concatenate(sums_chunks)
    del sums_chunks # Free memory
    
    all_sums.sort()
    
    # Find max frequency
    # np.unique on sorted array is fast
    unique_sums, counts = np.unique(all_sums, return_counts=True)
    
    max_count = np.max(counts)
    max_idx = np.argmax(counts)
    best_N = int(unique_sums[max_idx])
    
    # Reconstruct pairs
    powerfuls_set = set(P.tolist())
    pairs = []
    # P is sorted
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([int(a), int(b)])
            
    pairs.sort()
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}
