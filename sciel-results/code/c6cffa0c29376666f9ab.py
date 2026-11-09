import numpy as np
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # Generate powerful numbers up to L
    # L = 10^8 seems safe for memory/time
    L = 10**8
    
    # b^3 * a^2 <= L
    # b up to L^(1/3)
    max_b = int(L ** (1/3)) + 1
    powerful = set()
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            powerful.add(a * a * b3)
    
    P = np.array(sorted(powerful), dtype=np.int64)
    # P size approx 10^4
    
    # Check time
    if time.time() > deadline - 10:
         # Fallback to small search
         pass # We are likely fine here, generation is fast
    
    # Compute sums a+b for a <= b
    # We need to be careful with memory.
    # Number of pairs ~ 5 * 10^7
    # Creating a list of sums might be slow in Python.
    # Use NumPy broadcasting?
    # P[:, None] + P[None, :] creates a 10^4 x 10^4 matrix = 10^8 elements.
    # 800 MB.
    # Then take upper triangle.
    
    # Alternative:
    # Loop over a in P, add a to all b >= a.
    # This is 10^4 loops, each adding 10^4 elements.
    # Vectorized addition is fast.
    
    # Let's try to construct the sums array in chunks to avoid huge temporary arrays if needed,
    # but 800MB is likely okay for 120s if memory allows.
    # However, np.bincount needs a 1D array.
    
    # Let's generate sums in a list of arrays and concatenate?
    # Or just use the full matrix and flatten?
    
    # Matrix approach:
    # Sums = P[:, None] + P[None, :]
    # This is symmetric. We only need a <= b.
    # Mask = np.triu(np.ones((len(P), len(P)), dtype=bool))
    # sums = Sums[Mask]
    
    # This creates a 10^8 element array, then filters to 5*10^7.
    # Memory: 800MB + 400MB.
    
    # Let's try a more memory efficient way:
    # Iterate over a.
    # sums_list = []
    # for i, a in enumerate(P):
    #     sums_list.append(P[i:] + a)
    # sums = np.concatenate(sums_list)
    
    # This creates 10^4 arrays of decreasing size.
    # Total elements 5*10^7.
    # Concatenation might be slow but memory is managed.
    
    # Let's estimate time for concatenation of 50M ints.
    # Should be < 1s.
    
    sums_list = []
    for i in range(len(P)):
        # P[i:] + P[i]
        # This creates a new array
        sums_list.append(P[i:] + P[i])
        
    if time.time() > deadline - 30:
        # Fallback if too slow? Unlikely.
        pass
        
    sums = np.concatenate(sums_list)
    
    # Find max count
    # Max sum is P[-1] + P[-1] = 2 * L
    max_sum = int(2 * L)
    
    # bincount
    # minlength must be max_sum + 1
    # 2 * 10^8 + 1
    # int64 array of 200M elements = 1.6 GB.
    # This might be too much memory.
    
    # Alternative: use np.unique
    # unique_sums, counts = np.unique(sums, return_counts=True)
    # This sorts 50M elements.
    # Sorting 50M ints takes ~1-2s.
    # Memory: 400MB for sums + 400MB for unique_sums + 400MB for counts?
    # Should be okay.
    
    unique_sums, counts = np.unique(sums, return_counts=True)
    
    if time.time() > deadline - 20:
         # Find best so far?
         # We have counts.
         pass
         
    max_count = np.max(counts)
    best_N = unique_sums[np.argmax(counts)]
    
    # Find pairs
    pairs = []
    # Iterate P to find a,b such that a+b = best_N
    # Use set for fast lookup?
    # P is sorted.
    # Two pointers?
    # Or just loop a in P, check if best_N - a in P.
    
    P_set = set(P)
    for a in P:
        b = best_N - a
        if b < a:
            continue
        if b in P_set:
            pairs.append([int(a), int(b)])
            
    return {'N': int(best_N), 'pairs': pairs}
