"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from bisect import bisect_left, bisect_right
from collections import defaultdict

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form with sieve-based square-free check."""
    powerful = set()
    powerful.add(1)
    
    max_b = int(round(limit ** (1/3))) + 2
    
    # Precompute square-free status using sieve-like approach
    if max_b >= 1:
        sq_free = [True] * (max_b + 1)
        sq_free[0] = False
        if max_b >= 1:
            sq_free[1] = True
        for p in range(2, int(math.isqrt(max_b)) + 1):
            if sq_free[p]:
                p2 = p * p
                for m in range(p2, max_b + 1, p2):
                    sq_free[m] = False
    else:
        sq_free = [False] * (max_b + 1) if max_b >= 0 else []
    
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
    deadline = time.time() + 145  # safety margin before 150s limit
    start_time = time.time()
    
    # Phase 1: Generate powerful numbers up to a large limit
    limit = 500_000_000  # 5 * 10^8, based on measured improvement
    
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    arr = np.array(powerful, dtype=np.int64)
    
    # Phase 2: Count representations using chunked numpy approach
    # For each a, valid b must satisfy b >= a and b <= limit - a
    # Sum = a + b. We batch multiple a's together.
    
    sum_counts = defaultdict(int)
    
    # Only a values up to limit//2 can form pairs with b >= a
    valid_mask = arr <= limit // 2
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    a_vals = arr[valid_indices]
    
    # Compute j_start and j_end for each a using searchsorted
    j_starts = np.searchsorted(arr, a_vals, side='left')
    j_ends = np.searchsorted(arr, limit - a_vals, side='right')
    
    # Filter out invalid ranges
    valid_range = j_starts < j_ends
    if not np.any(valid_range):
        return {'N': 2, 'pairs': [[1, 1]]}
    
    valid_idx_filtered = valid_indices[valid_range]
    j_starts_f = j_starts[valid_range]
    j_ends_f = j_ends[valid_range]
    a_vals_f = a_vals[valid_range]
    
    n_filtered = len(valid_idx_filtered)
    
    # Process in chunks to manage memory and time
    # Each chunk: for a group of a's, collect all sums and count
    chunk_size = 200  # number of a values per chunk
    
    for chunk_start in range(0, n_filtered, chunk_size):
        if time.time() > deadline - 30:
            break
        
        chunk_end = min(chunk_start + chunk_size, n_filtered)
        
        all_sums = []
        for idx in range(chunk_start, chunk_end):
            a = a_vals_f[idx]
            js = j_starts_f[idx]
            je = j_ends_f[idx]
            if js >= je:
                continue
            b_vals = arr[js:je]
            sums = a + b_vals
            all_sums.append(sums)
        
        if not all_sums:
            continue
        
        flat_sums = np.concatenate(all_sums)
        unique_sums, counts = np.unique(flat_sums, return_counts=True)
        
        for s, c in zip(unique_sums, counts):
            sum_counts[int(s)] += int(c)
    
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
