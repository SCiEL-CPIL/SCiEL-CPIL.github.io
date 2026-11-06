"""Vectorized bincount-based powerful-number pair counting with larger L."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerfuls(limit):
    """Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    """
    powerfuls = set()
    # b must be square-free and b^3 <= limit
    max_b = int(limit ** (1/3)) + 2
    # Precompute primes up to sqrt(max_b) for square-free check
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        
        b3 = b ** 3
        if b3 > limit:
            break
        
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    return powerfuls

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Strategy: Use a larger limit L to find more powerful numbers.
    # L = 10^9 allows ~30,000 powerful numbers.
    # Pairwise sums: ~30k * 30k / 2 = 450 million pairs.
    # np.bincount on 450M integers is fast (memory ~3.6GB for int64, or less if int32).
    # We need to ensure memory is sufficient. 450M * 8 bytes = 3.6 GB.
    # If memory is tight, we can reduce L or use int32 if sums < 2^31 (which is ~2*10^9, so L=10^9 is safe for int32 sums up to 2*10^9).
    # Actually, sums can be up to 2*10^9, which fits in int32 (max ~2.1*10^9).
    # So we can use int32 for the sums array to save memory.
    
    L = 10**9
    
    # Check time
    if time.time() > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    powerfuls = _generate_powerfuls(L)
    powerfuls_sorted = np.array(sorted(powerfuls), dtype=np.int64)
    powerfuls_set = set(powerfuls_sorted.tolist())
    n = len(powerfuls_sorted)
    
    if time.time() > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Compute all pairwise sums a_i + a_j for i <= j
    # To avoid O(n^2) memory for the full matrix, we can use broadcasting and ravel?
    # n=30k -> 30k x 30k matrix is 900M elements. 900M * 8 bytes = 7.2 GB. Too much.
    # We need a more memory-efficient way or smaller L.
    
    # Let's try L = 5 * 10^8.
    # n ~ 15k. Pairs ~ 112M. 112M * 8 bytes = 900 MB. This is feasible.
    # Or L = 2 * 10^8. n ~ 7k. Pairs ~ 24M. Very safe.
    # The previous program used L=2*10^7 (n~1.5k) and got 4.13.
    # Inspiration 1 used L=10^8 (n~5k) and got 5.93.
    # Let's try L=10^8 with the bincount method which is simpler and potentially faster than FFT for small n.
    
    L = 10**8
    powerfuls = _generate_powerfuls(L)
    powerfuls_sorted = np.array(sorted(powerfuls), dtype=np.int64)
    powerfuls_set = set(powerfuls_sorted.tolist())
    n = len(powerfuls_sorted)
    
    if time.time() > deadline - 10:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Compute upper triangular sums
    # Using np.triu_indices
    i, j = np.triu_indices(n, k=0)
    sums = powerfuls_sorted[i] + powerfuls_sorted[j]
    
    # Count occurrences of each sum
    # sums can be up to 2*10^8, so bincount size is 2*10^8 + 1
    # This is 200M elements. 200M * 8 bytes = 1.6 GB.
    # If memory is an issue, we can use a dictionary or sort+count.
    # np.bincount is very fast.
    
    t0 = time.time()
    counts = np.bincount(sums, minlength=2 * L + 1)
    bin_time = time.time() - t0
    
    if time.time() > deadline - 5:
        # Fallback: just pick a random N with high count from top few
        # But we need pairs.
        # Let's just return a valid small N if we run out of time
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find N with max count
    # We need to be careful: counts[N] is the number of pairs (a,b) with a<=b and a+b=N.
    # This is exactly what we want.
    max_idx = np.argmax(counts)
    best_N = int(max_idx)
    best_count = int(counts[best_N])
    
    # Extract pairs for best_N
    pairs = []
    for a in powerfuls_sorted:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
            
    # Verify count matches
    # It should match best_count.
    
    # Check top 10 candidates to be safe
    top_k = 10
    # Get top k indices
    # Since counts is large, we can use argpartition or sort
    # But we only need a few.
    # Let's just check the top 10 by value
    top_indices = np.argsort(counts)[::-1][:top_k]
    
    best_final_N = best_N
    best_final_pairs = pairs
    best_final_count = len(pairs)
    
    for idx in top_indices:
        cand_N = int(idx)
        cand_pairs = []
        for a in powerfuls_sorted:
            if a > cand_N // 2:
                break
            b = cand_N - a
            if b in powerfuls_set:
                cand_pairs.append([a, b])
        if len(cand_pairs) > best_final_count:
            best_final_count = len(cand_pairs)
            best_final_N = cand_N
            best_final_pairs = cand_pairs
            
    if time.time() > deadline - 2:
        return {'N': best_final_N, 'pairs': best_final_pairs}
        
    return {'N': best_final_N, 'pairs': best_final_pairs}

# EVOLVE-BLOCK-END
