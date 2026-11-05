"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict
from bisect import bisect_left, bisect_right
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form with optimized square-free check."""
    powerful = set()
    powerful.add(1)
    max_b = int(round(limit ** (1/3))) + 2
    
    # Precompute square-free status using sieve-like approach
    # sq_free[i] = True if i is square-free
    sq_free = [True] * (max_b + 1)
    sq_free[0] = False
    # For each prime p, mark multiples of p^2 as not square-free
    for p in range(2, int(math.isqrt(max_b)) + 1):
        if sq_free[p]:  # p is prime (not marked by smaller prime squares)
            p2 = p * p
            for m in range(p2, max_b + 1, p2):
                sq_free[m] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    deadline = time.time() + 115  # safety margin before 150s limit
    start_time = time.time()
    
    # Phase 1: Generate powerful numbers up to a large limit
    limit = 80_000_000  # 8 * 10^7
    
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 2: Count representations using efficient numpy-based approach
    arr = np.array(powerful, dtype=np.int64)
    
    # We'll accumulate counts using a dictionary
    sum_counts = defaultdict(int)
    
    # Process a values in order; for each a, find valid b range
    for i in range(K):
        a = arr[i]
        if a > limit // 2:
            break
        # b must be >= a and <= limit - a
        j_start = bisect_left(arr, a)
        j_end = bisect_right(arr, limit - a)
        if j_start >= j_end:
            continue
        
        # Get the b values and compute sums
        b_vals = arr[j_start:j_end]
        sums = a + b_vals
        
        # Use np.unique to count occurrences of each sum
        unique_sums, counts = np.unique(sums, return_counts=True)
        for s, c in zip(unique_sums, counts):
            sum_counts[int(s)] += int(c)
        
        # Time check: if we're running low, stop
        if time.time() > deadline - 40:
            break
    
    if not sum_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the best N
    best_N = max(sum_counts.keys(), key=lambda n: sum_counts[n])
    best_count = sum_counts[best_N]
    
    # Generate the actual pairs for best_N
    pairs = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
    
    # Verify
    if len(pairs) != best_count:
        best_count = len(pairs)
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
